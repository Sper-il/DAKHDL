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

## 5. Output mỗi lần train
```
results/runs/<ts>_<branch>/
├── config.yaml   # branch + train + data + model đã dùng
├── tb/           # log TensorBoard (loss/train, loss/val, f1/val)
models/checkpoints/<branch>/
├── best.pt       # full checkpoint tốt nhất
├── best.state.pt # chỉ state_dict model tốt nhất
└── last.pt       # checkpoint epoch cuối
```

## 6. Tiền kiểm bắt buộc
- Overfit 1 batch nhỏ (`--sanity`) phải giảm loss về gần 0 trước khi train full.
- Chạy thử 1 epoch trên subset để ước lượng thời gian và VRAM.
- Seed cố định 42 cho Python, NumPy, PyTorch, ghi trong config.
- Val luôn dùng transform không augment, `model.eval()` + `torch.no_grad()`.

## 7. Xử lý sự cố
| Hiện tượng | Nguyên nhân thường gặp | Cách xử lý |
|------------|------------------------|------------|
| Loss NaN | LR quá cao, thiếu clipping, AMP overflow | Giảm LR, giữ `gradient_clipping=1.0`, tắt AMP thử |
| Train/val gap lớn | Overfit, mất cân bằng lớp | Tăng augment, weight decay, dropout, dùng class weight |
| Không cải thiện F1 val | LR sai, scheduler sai, data leakage | Kiểm tra split theo speaker, xem confusion matrix |
| Hết VRAM | Batch quá lớn khi train 2 nhánh | Train riêng từng nhánh, giảm batch 64→32 (mặt), 32→16 (tiếng) |
