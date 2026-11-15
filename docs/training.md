# Training — Quy trình huấn luyện Emotion Fusion

> Tài liệu vận hành train 2 nhánh. Đồng bộ với `configs/train_config.yaml` và `src/training/`.

## 1. Nguyên tắc
- Train riêng từng nhánh (face, audio), không train chung để tránh lệch tốc độ học.
- Mỗi lần train sinh một folder `results/runs/<ts>_<branch>/` mới, không ghi đè.
- Best checkpoint chọn theo `val_f1_macro`, không theo loss train.
- Mọi lần train đều lưu `config.yaml` để reproduce từ seed + config.

## 2. Config huấn luyện
File `configs/train_config.yaml`:

```yaml
seed: 42
optimizer: AdamW
learning_rate: 0.0003
weight_decay: 0.0001
epochs: 30
scheduler: CosineAnnealingLR
early_stopping:
  patience: 7
  monitor: val_f1_macro
gradient_clipping: 1.0
mixed_precision: true
checkpoint_dir: models/checkpoints
```

Batch size nằm trong `configs/data_config.yaml`: mặt 64, tiếng 32. Sanity check dùng batch 8.

## 3. Trainer
Mã nguồn `src/training/trainer.py::Trainer`:

- `train_one_epoch(epoch)`: forward, `CrossEntropyLoss`, AMP khi có CUDA, clip grad `max_norm=1.0`, step scheduler `CosineAnnealingLR`.
- `validate()`: `model.eval()` + `torch.no_grad()`, trả về `{loss, acc, f1_macro}`.
- `save_checkpoint(epoch, name)`: lưu `{epoch, model, opt, sched, best_f1}` vào `models/checkpoints/<branch>/`.
- `fit()`: lặp epoch, lưu `last.pt` mỗi epoch, lưu `best.pt` + `best.state.pt` khi `f1_macro` cải thiện, early stopping sau `patience=7` epoch không cải thiện, log TensorBoard vào `run_dir/tb/`.

## 4. Lệnh chạy
```bash
# Sanity check 2 epoch, overfit 1 batch nhỏ
python -m src.training --branch face \
  --train-csv data/splits/face_train.csv \
  --val-csv data/splits/face_val.csv --sanity

python -m src.training --branch audio \
  --train-csv data/splits/audio_train.csv \
  --val-csv data/splits/audio_val.csv --sanity

# Train full
python -m src.training --branch face \
  --train-csv data/splits/face_train.csv \
  --val-csv data/splits/face_val.csv

python -m src.training --branch audio \
  --train-csv data/splits/audio_train.csv \
  --val-csv data/splits/audio_val.csv
```

Flag `--epochs N` ghi đè số epoch, `--config/--data-config/--model-config` ghi đè đường dẫn yaml.

## 5. Tìm kiếm siêu tham số (4 model × 6 tham số × 6–7 giá trị)

Mỗi nhánh thử **4 model variants** × **6 siêu tham số** (mỗi tham số 6–7 giá trị) = random search **160 trial/nhánh**, so sánh theo `val_f1_macro`, chọn cấu hình tốt nhất.

> Chi tiết đầy đủ (từng bước, lệnh chạy, output, DoD): **`docs/plan_remaining.md` — Bước 1.3 → 1.6**.

**Khai báo không gian tìm kiếm** — `configs/tuning_space.yaml`:
```yaml
face:
  model_variants:
    - {name: resnet18_gray, backbone: resnet18, pretrained: true}
    - {name: resnet18_gray_nopretrain, backbone: resnet18, pretrained: false}
    - {name: mobilenetv2_gray, backbone: mobilenet_v2, pretrained: true}
    - {name: small_cnn, backbone: small_cnn, pretrained: false}
  hyperparameters:
    learning_rate: [0.0001, 0.0003, 0.001, 0.003, 0.01, 0.00003]
    weight_decay: [0.00001, 0.0001, 0.001, 0.01, 0.00005, 0.0005]
    batch_size: [32, 64, 128, 16, 96, 256]
    dropout: [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    label_smoothing: [0.0, 0.05, 0.1, 0.15, 0.2, 0.3]
    class_weight_power: [0.0, 0.25, 0.5, 0.75, 1.0, 1.5]
  epochs: 12
  patience: 4
  objective: val_f1_macro
```

**Chạy** — `src/training/tuner.py`:
```bash
python -m src.training tune --branch face \
  --space configs/tuning_space.yaml --trials 40

python -m src.training tune --branch audio \
  --space configs/tuning_space.yaml --trials 40
```

**6 siêu tham số được chỉnh:**

| # | Tham số | Mặt — giá trị thử | Tiếng — giá trị thử |
|---|---------|-------------------|--------------------|
| 1 | `learning_rate` | 1e-4 / 3e-4 / 1e-3 / 3e-3 / 1e-2 / 3e-5 | (giống mặt) |
| 2 | `weight_decay` | 1e-5 / 1e-4 / 1e-3 / 1e-2 / 5e-5 / 5e-4 | (giống mặt) |
| 3 | `batch_size` | 16 / 32 / 64 / 96 / 128 / 256 | 8 / 16 / 32 / 64 / 96 / 128 |
| 4 | `dropout` | 0,0 / 0,1 / 0,2 / 0,3 / 0,4 / 0,5 | (giống mặt) |
| 5 | `label_smoothing` | 0,0 / 0,05 / 0,1 / 0,15 / 0,2 / 0,3 | (giống mặt) |
| 6 | `class_weight_power` | 0,0 / 0,25 / 0,5 / 0,75 / 1,0 / 1,5 | `spec_augment_prob`: 0,0 / 0,1 / 0,2 / 0,3 / 0,4 / 0,5 |

**Output:**
```
results/tuning/<branch>/
└── trials.csv   # trial_id, model_variant, learning_rate, weight_decay, batch_size, dropout,
                 # label_smoothing, class_weight_power, val_f1_macro, val_acc, train_time_s, status
models/checkpoints/<branch>/
├── tuning_best.pt           # checkpoint của trial thắng
└── tuning_best_config.yaml  # cấu hình thắng, dùng để train full lại
```

**Quy tắc chọn model:**
- Chọn theo **`val_f1_macro` cao nhất** trên tập val.
- **Không** dùng test set để chọn — chỉ chạy test 1 lần sau khi đã chốt model.
- Sau khi chốt, train full 30 epoch với config thắng để ra checkpoint chính thức.
- Ghi bảng so sánh baseline vs tuned vào `docs/progress_status.md` và `docs/decisions.md`.
- Nâng lên **Optuna** nếu có thời gian — random search là phương án tốt nhất với ngân sách GPU hạn chế (6^6 ≈ 46.656 tổ hợp không thể grid search đầy đủ).

## 6. Output mỗi lần train
```
results/runs/<ts>_<branch>/
├── config.yaml   # branch + train + data + model đã dùng
├── tb/           # log TensorBoard (loss/train, loss/val, f1/val)
models/checkpoints/<branch>/
├── best.pt       # full checkpoint tốt nhất
├── best.state.pt # chỉ state_dict model tốt nhất
└── last.pt       # checkpoint epoch cuối
```

## 7. Tiền kiểm bắt buộc
- Overfit 1 batch nhỏ (`--sanity`) phải giảm loss về gần 0 trước khi train full.
- Chạy thử 1 epoch trên subset để ước lượng thời gian và VRAM.
- Seed cố định 42 cho Python, NumPy, PyTorch, ghi trong config.
- Val luôn dùng transform không augment, `model.eval()` + `torch.no_grad()`.
- **⚠️ `--sanity` ghi đè `best.pt`** của nhánh đó. Sau khi chạy sanity phải train thật lại, nếu không checkpoint sẽ là của lần sanity và không dùng được để eval.

## 8. Xử lý sự cố
| Hiện tượng | Nguyên nhân thường gặp | Cách xử lý |
|------------|------------------------|------------|
| Loss NaN | LR quá cao, thiếu clipping, AMP overflow | Giảm LR, giữ `gradient_clipping=1.0`, tắt AMP thử |
| Train/val gap lớn | Overfit, mất cân bằng lớp | Tăng augment, weight decay, dropout, dùng class weight |
| Không cải thiện F1 val | LR sai, scheduler sai, data leakage | Kiểm tra split theo speaker, xem confusion matrix |
| Hết VRAM | Batch quá lớn khi train 2 nhánh | Train riêng từng nhánh, giảm batch 64→32 (mặt), 32→16 (tiếng) |
