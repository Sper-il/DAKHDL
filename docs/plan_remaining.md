# Plan công việc còn lại — Giao cho thành viên

> Cập nhật: 2026-11-15
> Mỗi bước ghi rõ: **Làm file gì** → **Chức năng** → **Làm thêm sau được gì**
>
> Trạng thái repo: DATA ✅ · MODELS ✅ · TRAINING (code ✅, chưa train full) · EVALUATION ⏳ · VISUALIZE ⏳ · DEMO ⏳

---

## Tổng quan 4 bước

| Bước | Làm gì | Kết quả | Thời gian |
|------|--------|---------|-----------|
| **1** | Train + tìm model tối ưu | `best.pt` cho mặt và tiếng | 4–6 ngày GPU |
| **2** | Đánh giá (EVAL) | `metrics.json` có 3 nhánh | 2 ngày |
| **3** | Vẽ báo cáo (VISUALIZE) | `report.html` offline | 1–2 ngày |
| **4** | Demo 1 lệnh (DEMO) | `result.json` + `index.html` | 1 ngày |

> Thứ tự bắt buộc **1 → 2 → 3 → 4**. Bước 2 và 4 cần checkpoint tốt từ bước 1; bước 3 cần `metrics.json` từ bước 2.
> Bước 2, 3, 4 chia được cho 3 người làm song song (sau khi bước 1 xong).

---

## BƯỚC 1 — Train + tìm model tối ưu

**Mục tiêu:** có checkpoint tốt nhất cho mặt và tiếng.

### 1.1 — Sanity check (30 phút)

**Làm file gì:** Không tạo file. Chạy lệnh có sẵn.

```bash
python -m src.training --branch face \
  --train-csv data/splits/face_train.csv \
  --val-csv data/splits/face_val.csv --sanity

python -m src.training --branch audio \
  --train-csv data/splits/audio_train.csv \
  --val-csv data/splits/audio_val.csv --sanity
```

**Chức năng:** `--sanity` lấy 32 mẫu train + 16 mẫu val, train 2 epoch. Kiểm tra nhanh: đường dẫn CSV, đọc ảnh/mel, shape đúng model, backward pass chạy, checkpoint ghi được.

**Đạt khi:** loss giảm, không lỗi shape.

**Làm thêm sau được gì:** nếu lỗi shape → đối chiếu `docs/models.md`. Lỗi data → chạy lại `python -m src.data.make_splits`.

> ⚠️ `--sanity` **ghi đè** `best.pt`. Sau bước này phải train thật lại.

### 1.2 — Train baseline (2–4 giờ GPU)

**Làm file gì:** Không tạo file. Chạy trainer có sẵn.

```bash
python -m src.training --branch face \
  --train-csv data/splits/face_train.csv \
  --val-csv data/splits/face_val.csv

python -m src.training --branch audio \
  --train-csv data/splits/audio_train.csv \
  --val-csv data/splits/audio_val.csv
```

**Chức năng:** `src/training/trainer.py` train 30 epoch, AdamW lr 3e-4, early stopping `patience=7` theo `val_f1_macro`. Lưu `best.pt` vào `models/checkpoints/<branch>/`.

**Đạt khi:** `best_f1_macro` ≥ 0,60 mỗi nhánh. Ghi lại số này làm baseline để so sánh.

**Làm thêm sau được gì:**
- F1 thấp → bật `class_weights` trong `configs/model_config.yaml`; lớp mới rất ít mẫu (mặt boredom 358, excited 600; tiếng excited 401) nên tăng `class_weight`.
- 7 lớp gốc tụt → ảnh extra khác domain FER2013, xem bước 1.6.
- Train thêm epoch nếu loss train còn giảm mạnh khi val đã đứng.

### 1.3 — Chuẩn bị không gian tìm kiếm (T0 + T0b)

**Làm file gì:**
- Tạo mới `configs/tuning_space.yaml`
- Sửa `src/models/face_cnn.py`, `src/models/audio_cnn.py`, `src/models/architecture.py`

**Chức năng:**

| File | Việc |
|------|------|
| `configs/tuning_space.yaml` | Khai báo **4 model variants** + **6 siêu tham số**, mỗi tham số **6–7 giá trị** |
| `src/models/face_cnn.py` | Thêm `build_face_cnn(backbone=...)` cho resnet18 / mobilenet_v2 / small_cnn |
| `src/models/audio_cnn.py` | Cho phép đổi `n_blocks`, `channels` |
| `src/models/architecture.py` | `build_model(branch, variant=...)` chọn kiến trúc theo config |

**Model variants:**

| Nhánh | Variant | Ghi chú |
|-------|---------|---------|
| Mặt | `resnet18_gray` | Baseline hiện tại |
| Mặt | `resnet18_gray_nopretrain` | Kiểm chứng pretrained có giúp không |
| Mặt | `mobilenetv2_gray` | Nhẹ hơn → hợp latency ≤ 200 ms |
| Mặt | `small_cnn` | CNN tự viết, đối chứng |
| Tiếng | `cnn_4blocks` | Baseline hiện tại |
| Tiếng | `cnn_3blocks_wide` | Ít layer, nhiều capacity |
| Tiếng | `cnn_5blocks` | Tham khảo VGG-style |
| Tiếng | `cnn_4blocks_deep` | Dropout cao |

**6 siêu tham số:**

| # | Tham số | Mặt — 6–7 giá trị | Tiếng — 6–7 giá trị |
|---|---------|-------------------|--------------------|
| 1 | `learning_rate` | 1e-4 / 3e-4 / 1e-3 / 3e-3 / 1e-2 / 3e-5 | (giống mặt) |
| 2 | `weight_decay` | 1e-5 / 1e-4 / 1e-3 / 1e-2 / 5e-5 / 5e-4 | (giống mặt) |
| 3 | `batch_size` | 16 / 32 / 64 / 96 / 128 / 256 | 8 / 16 / 32 / 64 / 96 / 128 |
| 4 | `dropout` | 0,0 / 0,1 / 0,2 / 0,3 / 0,4 / 0,5 | (giống mặt) |
| 5 | `label_smoothing` | 0,0 / 0,05 / 0,1 / 0,15 / 0,2 / 0,3 | (giống mặt) |
| 6 | `class_weight_power` | 0,0 / 0,25 / 0,5 / 0,75 / 1,0 / 1,5 | `spec_augment_prob`: 0,0 / 0,1 / 0,2 / 0,3 / 0,4 / 0,5 |

**Đạt khi:** YAML hợp lệ, mỗi tham số có 6–7 giá trị, có 4 variant mỗi nhánh, và `pytest tests/unit -q` vẫn pass (gọi `build_model("face", num_classes=10)` không có `variant` phải ra model như cũ).

**Làm thêm sau được gì:** thêm `efficientnet_b0` cho mặt; thêm `n_mels` ∈ {64, 96, 128, 160} cho tiếng.

### 1.4 — Viết engine tuning (T1 + T2)

**Làm file gì:**
- Tạo mới `src/training/tuner.py`
- Sửa `src/training/__main__.py` — thêm lệnh `tune`

**Chức năng của `tuner.py`:**

| Hàm | Việc |
|-----|------|
| `generate_combinations(space, mode)` | Sinh tổ hợp siêu tham số. `mode="random"` lấy ngẫu nhiên N tổ hợp (seed 42) |
| `run_trial(model, trial_id, hparams)` | Train 1 tổ hợp, trả `val_f1_macro` + `val_acc` |
| `tune(branch, space, n_trials)` | Chạy hết trial, lưu `trials.csv`, trả trial tốt nhất |

**Chi tiết:**
- Dùng **random search** vì 6^6 ≈ 46.656 tổ hợp — grid search đầy đủ không khả thi.
- Tuning dùng `epochs=12`, `patience=4` (ngắn hơn train full) để tiết kiệm GPU.
- Ghi từng trial vào `results/tuning/<branch>/trials.csv`:
  ```
  trial_id,model_variant,learning_rate,weight_decay,batch_size,dropout,label_smoothing,class_weight_power,val_f1_macro,val_acc,train_time_s,status
  ```
- Chỉ giữ checkpoint của trial thắng: `models/checkpoints/<branch>/tuning_best.pt` + `tuning_best_config.yaml`.
- Cần `--resume` để bỏ qua trial đã chạy khi bị crash.

**Lệnh:**
```bash
python -m src.training tune --branch face \
  --space configs/tuning_space.yaml --trials 40

python -m src.training tune --branch audio \
  --space configs/tuning_space.yaml --trials 40
```

**Đạt khi:** lệnh `train` cũ vẫn chạy y như trước; lệnh `tune` sinh `trials.csv` có ≥ 40 dòng, cột `val_f1_macro` điền hết.

**Làm thêm sau được gì:** nâng lên **Optuna** (tự chọn tổ hợp có triển vọng → ít trial hơn); Hyperband (bỏ trial kém sớm); chạy song song 2 trial nếu có 2 GPU.

### 1.5 — Quét + train lại với config tối ưu (T3–T5)

**Làm file gì:** Không tạo file. Chạy lệnh.

**Quy trình:**
1. Chạy tune cho mặt (160 trial) và tiếng (160 trial).
2. Lấy top-10 mỗi nhánh theo `val_f1_macro`.
3. Train full 30 epoch với config thắng:
```bash
python -m src.training --branch face \
  --train-csv data/splits/face_train.csv \
  --val-csv data/splits/face_val.csv \
  --config models/checkpoints/face/tuning_best_config.yaml

python -m src.training --branch audio \
  --train-csv data/splits/audio_train.csv \
  --val-csv data/splits/audio_val.csv \
  --config models/checkpoints/audio/tuning_best_config.yaml
```

**Đạt khi:** `best.pt` mới có `best_f1_macro` **cao hơn baseline** ở 1.2. Ghi cả 2 số để so sánh.

**Làm thêm sau được gì:**
- Nếu 160 trial quá lâu → giảm `--trials` còn 20 (vẫn 80 trial/nhánh, đủ tin cậy).
- Train 3 seed với config thắng → báo cáo trung bình + độ lệch.
- **Ensemble** 3 checkpoint tốt nhất (average probability) — thường cộng thêm 1–3 điểm accuracy.

### 1.6 — Lập bảng so sánh (T6)

**Làm file gì:** Không tạo file. Ghi kết quả vào `docs/progress_status.md`.

**Bảng phải có:**

| Run | Variant | lr | wd | bs | dropout | ls | cw | val F1 | val acc | test F1 | test acc |
|-----|---------|----|----|----|---------|----|----|---------|---------|---------|---------|
| Baseline | resnet18_gray | 3e-4 | 1e-4 | 64 | 0,3 | 0 | 0 | 0,xxx | 0,xxx | — | — |
| Tuning best | xxx | ... | ... | ... | ... | ... | ... | 0,xxx | 0,xxx | 0,xxx | 0,xxx |

**Làm thêm sau được gì:** thêm bảng top-10 trial mỗi nhánh vào `docs/decisions.md`; vẽ đồ thị importance của từng siêu tham số bằng `scripts/plot_tuning.py`.

---

## BƯỚC 2 — Đánh giá (EVAL)

**Mục tiêu:** đo được mặt, tiếng, fusion trên test set.

### 2.1 — Viết `src/evaluation/metrics.py`

**Làm file gì:** Mở file rỗng có sẵn `src/evaluation/metrics.py` và viết vào.

**Chức năng:** thư viện hàm tính metric, không chạy gì.

| Hàm | Input | Output |
|-----|-------|--------|
| `evaluate_branch(model, loader, device)` | model + loader test | `accuracy`, `f1_macro`, `confusion_matrix` (10×10), `inference_time_ms` |
| `evaluate_fusion(face_model, audio_model, loader_pairs, w_face, w_audio)` | 2 model + loader cặp | như trên + `w_face`, `w_audio`, `e2e_latency_ms` |

- Dùng `sklearn.metrics`: `accuracy_score`, `f1_score(average="macro")`, `confusion_matrix`.
- Bắt buộc: đo latency có `torch.cuda.synchronize()`.
- Bắt buộc: `evaluate_fusion` assert `yf == ya`.
- Dùng `fuse_probs` có sẵn từ `src/models/fusion.py` — đừng viết lại logic fusion.

**Đạt khi:** gọi được trên loader test, ra đủ 4 trường, confusion đúng 10×10.

**Làm thêm sau được gì:** `per_class_f1()` (cần cho báo cáo vì 3 lớp mới ít mẫu); `top_k_accuracy()`; `evaluate_by_source()` (breakdown RAVDESS/CREMA-D/ASVP để kiểm tra domain shift).

### 2.2 — Viết `src/evaluation/__main__.py`

**Làm file gì:** Mở file rỗng có sẵn `src/evaluation/__main__.py` và viết vào.

**Chức năng:** CLI entry point. Đọc checkpoint → chạy test → ghi `metrics.json`.

```bash
python -m src.evaluation \
  --face-ckpt models/checkpoints/face/best.pt \
  --face-test-csv data/splits/face_test.csv \
  --audio-ckpt models/checkpoints/audio/best.pt \
  --audio-test-csv data/splits/audio_test.csv \
  --run-dir results/runs/<ts>
```

- **Suy `num_classes` từ shape checkpoint** (`backbone.fc.weight` cho mặt, `head.5.weight` cho tiếng) — không hardcode 10.
- Đọc target từ `configs/eval_config.yaml`, không hardcode.
- Ghi `results/runs/<ts>/metrics.json`:
```json
{
  "time": "20261115-1900",
  "branches": {
    "face":  {"accuracy": 0.66, "target": 0.65, "passed": true, "f1_macro": 0.61},
    "audio": {"accuracy": 0.71, "target": 0.70, "passed": true, "f1_macro": 0.68},
    "fusion":{"accuracy": 0.74, "target": 0.70, "passed": true, "f1_macro": 0.66}
  },
  "e2e_latency_ms": 42.5
}
```

**Đạt khi:** `metrics.json` có đủ 3 nhánh, số khớp target trong config.

**Làm thêm sau được gì:** `--split val` để eval trên val mà không đụng test; ghi `git rev-parse HEAD` vào JSON để truy vết checkpoint.

### 2.3 — Ghép cặp face/audio cho fusion

**Làm file gì:** Bổ sung hàm vào `src/evaluation/metrics.py`.

**Vấn đề:** test mặt và test tiếng là 2 bộ mẫu **không cùng index**, không ghép theo vị trí được.

**Cách làm:** mỗi mẫu mặt ghép với 1 mẫu tiếng **cùng label**, lấy random `seed=42`. Lớp nào không đủ mẫu thì bỏ qua, ghi vào `error_analysis`.

**Bắt buộc:** ghi cách ghép vào `metrics.json` để tái tạo được:
```json
{"pairing": "random_same_label", "n_pairs": 4175, "seed": 42}
```

**Bắt buộc:** ghi rõ trong báo cáo đây là ghép cặp **giả lập**, không phải cùng một người — nếu không là sai lệch khoa học.

**Làm thêm sau được gì:** đo fusion trên cặp thật nếu tìm được dataset có cùng người; ablation so sánh `w_face` khác nhau trên cùng bộ cặp.

---

## BƯỚC 3 — Vẽ báo cáo (VISUALIZE)

**Mục tiêu:** 1 file `report.html` mở offline, đủ số liệu.

### 3.1 — Viết `src/evaluation/visualize.py`

**Làm file gì:** Mở file rỗng có sẵn `src/evaluation/visualize.py` và viết vào.

**Chức năng:** vẽ PNG từ kết quả eval.

| Hàm | Output |
|-----|--------|
| `plot_confusion_matrix(cm, labels, title)` | PNG ma trận 10×10, có số trong ô |
| `plot_loss_curve(run_dir)` | PNG loss train/val (đọc từ TensorBoard log) |

- matplotlib, backend `Agg` (không cần GUI).
- Lưu vào `results/runs/<ts>/plots/`.

**Đạt khi:** sinh được `confusion_face.png`, `confusion_audio.png`, `confusion_fusion.png`, `loss_curve.png`.

**Làm thêm sau được gì:** `plot_per_class_f1()` (thanh ngang F1 từng lớp); `plot_pr_curve()` / `plot_roc_curve()`; `plot_error_distribution()`.

### 3.2 — Viết `src/evaluation/visualize_html.py`

**Làm file gì:** Mở file rỗng có sẵn `src/evaluation/visualize_html.py` và viết vào.

**Chức năng:** đóng gói kết quả thành **1 file HTML duy nhất**, mở offline bằng double-click.

| Hàm | Việc |
|-----|------|
| `_img_b64(path, max_side=800)` | resize cạnh dài về 800, nén JPEG q80, nhúng base64 |
| `metrics_table(metrics)` | bảng nhánh / target / accuracy / F1 / ĐẠT-CHƯA ĐẠT |
| `build_report(run_dir, template_path)` | đọc JSON + ảnh, thay placeholder, ghi `report.html` |

- Template: `.cursor/skills/ai-fusion-workflow/templates/report.html`.
- Placeholder: `PROJECT`, `RUN_ID`, `COMMIT`, `TIME`, `CHECKPOINT`, `METRICS_TABLE`, `LOSS_CURVE_IMG`, `GOOD_SAMPLES`, `BAD_CASES`, `E2E_LATENCY_MS`.
- **Offline 100%:** không CDN, không font ngoài, CSS inline, ảnh base64.
- Số trong HTML phải khớp **100%** `metrics.json` — không hardcode, không sửa tay.

**6 nhóm nội dung bắt buộc trong `report.html`:**
1. Bảng metric pass/fail theo `configs/eval_config.yaml`
2. Đồ thị loss train/val
3. Confusion matrix 3 nhánh (10×10)
4. Mẫu tốt + 5 case tệ nhất
5. Latency end-to-end, throughput, VRAM nếu đo được
6. Limitations + next steps cụ thể

**Đạt khi:** copy sang máy không mạng → double-click vẫn hiện đủ bảng và ảnh; dung lượng < 10 MB.

**Làm thêm sau được gì:** tab "So sánh các run"; dark mode bằng CSS thuần; xuất bản `index.html` rút gọn cho demo (bước 4).

---

## BƯỚC 4 — Demo 1 lệnh (DEMO)

**Mục tiêu:** 1 lệnh ra cảm xúc dự đoán + JSON + HTML.

### 4.1 — Chuẩn bị input mẫu

**Làm file gì:** Sửa file có sẵn `src/data/make_demo.py`.

**Chức năng:** chọn 1 ảnh + 1 WAV đại diện từ tập test, lưu vào `data/processed/demo/`.

- Sinh `face_01.png` (48×48 grayscale) + `voice_01.wav` (16 kHz mono 3–5 giây).
- Mỗi mẫu < 5 MB.
- Chọn mẫu có **đúng label** để dễ kiểm chứng.

**Đạt khi:** `python -m src.data.make_demo` xong, có 2 file trong `data/processed/demo/`.

**Làm thêm sau được gì:** sinh nhiều bộ mẫu (`face_01..face_05`); làm bộ riêng cho từng lớp cảm xúc để trình diễn đa dạng.

### 4.2 — Viết `src/demo/__main__.py`

**Làm file gì:** Mở file rỗng có sẵn `src/demo/__main__.py` và viết vào.

**Chức năng:** CLI 1 lệnh — nhận 1 ảnh + 1 WAV → trả cảm xúc dự đoán.

```bash
python -m src.demo \
  --image data/processed/demo/face_01.png \
  --audio data/processed/demo/voice_01.wav \
  --face-ckpt models/checkpoints/face/best.pt \
  --audio-ckpt models/checkpoints/audio/best.pt \
  --w-face 0.5 --w-audio 0.5
```

- **Suy `num_classes` từ shape checkpoint** (`backbone.fc.weight` cho mặt, `head.5.weight` cho tiếng) — không hardcode 10.
- Label lấy từ `EMOTIONS` trong `src/models/fusion.py` (10 phần tử, thứ tự khớp cả 2 nhánh).
- Tiền xử lý: ảnh resize 48×48 grayscale, normalize mean/std 0,5; audio resample 16 kHz mono, pad/cắt 4 giây, mel 128×128.
- Có `--w-face` / `--w-audio` để chỉnh trọng số fusion.

**Output** — `results/demo/<ts>/result.json`:
```json
{
  "emotion": "happy",
  "confidence": 0.87,
  "probs_fused": [0.02, 0.01, 0.01, 0.87, 0.03, 0.05, 0.01, 0.00, 0.00, 0.00],
  "inference_time_ms": 42.0
}
```
- `probs_fused` đủ **10 phần tử**, tổng = 1.
- `index.html` bản rút gọn của `report.html`, mở offline.

**Đạt khi:** chạy được trên máy sạch sau `pip install -r requirements.txt`; JSON đủ 4 trường; `index.html` mở offline; latency ≤ 200 ms trên GPU lab.

**Làm thêm sau được gì:** `--realtime` (webcam + mic liên tục); `--no-audio` / `--no-face` (chạy được khi chỉ có 1 modality); export ONNX / Gradio UI.

---

## Việc phụ (làm khi rảnh tay)

| # | Việc | Làm gì | Ghi kết quả ở đâu |
|---|------|--------|-------------------|
| P1 | Dọn output cũ 7 lớp | Xoá hoặc chuyển `results/runs/20261109-1704/`, `results/demo/20261109-1706/` sang `results/archive/` | `docs/decisions.md` |
| P2 | Ablation ảnh extra | Train lại nhánh mặt chỉ với 7 lớp gốc FER2013, so sánh với bước 1.5 | `docs/progress_status.md` |
| P3 | Chốt trọng số fusion | Quét `w_face` ∈ {0,3 / 0,4 / 0,5 / 0,6 / 0,7} trên **val** | `docs/progress_status.md` (Decisions Log) |

> ⛔ P3 **cấm** tune trên test — sẽ là rò rỉ test set.

---

## Files cần viết / sửa

| File | Trạng thái | Bước |
|------|-----------|------|
| `configs/tuning_space.yaml` | Tạo mới | 1.3 |
| `src/training/tuner.py` | Tạo mới | 1.4 |
| `src/models/face_cnn.py` | Sửa có sẵn | 1.3 |
| `src/models/audio_cnn.py` | Sửa có sẵn | 1.3 |
| `src/models/architecture.py` | Sửa có sẵn | 1.3 |
| `src/training/__main__.py` | Sửa có sẵn | 1.4 |
| `src/evaluation/metrics.py` | **Rỗng có sẵn** | 2.1 |
| `src/evaluation/__main__.py` | **Rỗng có sẵn** | 2.2 |
| `src/evaluation/visualize.py` | **Rỗng có sẵn** | 3.1 |
| `src/evaluation/visualize_html.py` | **Rỗng có sẵn** | 3.2 |
| `src/data/make_demo.py` | Sửa có sẵn | 4.1 |
| `src/demo/__main__.py` | **Rỗng có sẵn** | 4.2 |

> `src/evaluation/__init__.py` và `src/demo/__init__.py` giữ placeholder docstring — không cần sửa.

---

## ⚙️ Quy tắc bắt buộc

1. **Code xong ≠ Module xong.** Viết xong code phải cập nhật docs trong cùng PR — theo rule trong `.cursor/skills/ai-fusion-workflow/SKILL.md`.
2. **Viết vào file rỗng có sẵn.** Mở file ra viết, đừng tạo file tên khác.
3. **File của bước chưa làm phải trống.** Chạy `python scripts/check_pending_modules.py` trước khi commit — lệnh này chặn việc tạo file rỗng rồi tưởng đã làm xong, và chặn việc sửa file chưa tới bước.
4. **Cập nhật `docs/progress_status.md`** sau mỗi bước: Module Log thêm dòng, Active Module cập nhật trạng thái, Last run ghi lệnh + kết quả thật.
5. **Không bịa số liệu.** Mọi con số phải truy được về 1 file output cụ thể (`metrics.json`, `trials.csv`, TensorBoard log, checkpoint).
6. **Tune chỉ trên val.** Tuyệt đối không dùng test set để chọn siêu tham số hay trọng số fusion.
7. **Không trích dẫn output cũ 7 lớp** — sai thang, sai schema.
8. **Chạy `pytest tests/unit -q` trước khi commit** — phải pass hết.
9. **Commit theo Conventional Commits**: `feat(tuning): ...`, `feat(models): ...`, `feat(eval): ...`, `feat(demo): ...`, `docs(tuning): ...`

---

## Phân công đề xuất

| Người | Bước | Lý do |
|-------|------|-------|
| **Thành viên 1** | 1.1 → 1.2 → 1.3 → 1.4 → 1.5 → 1.6 | Tốn GPU nhất, làm trước để unblock tất cả |
| **Thành viên 2** | 2.1 → 2.2 → 2.3 (+ P3) | Xử lý số liệu, không phụ thuộc GPU lâu |
| **Thành viên 3** | 3.1 → 3.2 | Làm sau bước 2, có thể viết `visualize.py` trước khi có metrics.json |
| **Thành viên 4** | 4.1 → 4.2 (+ P1) | Làm sau bước 1, độc lập với bước 2/3 |
| **Trưởng nhóm** | P2, review, commit, cập nhật `progress_status.md` | Việc tổng hợp |

> Nếu chỉ có **2–3 người**: gộp 3 + 4 cho 1 người, bước 2 cho 1 người, bước 1 cho 1 người.

---

## Timeline gợi ý

| Ngày | Việc |
|------|------|
| **1** | 1.1 sanity → 1.2 train baseline (khởi động sớm) · các thành viên khác đọc đặc tả trong `docs/` |
| **2** | 1.2 xong → 1.3 khai báo tuning space + mở rộng `src/models/` · 2.1 viết `metrics.py` · 3.1 viết `visualize.py` · 4.2 viết CLI demo |
| **3** | 1.4 viết `tuner.py` + lệnh `tune` → bắt đầu quét mặt · 2.2/2.3 chạy eval baseline · 4.1 chuẩn bị mẫu |
| **4** | Quét tiếng (song song nếu có 2 GPU) · 3.2 render report · 4.2 chạy demo |
| **5** | 1.5 train full với config tối ưu (2 nhánh) · 1.6 lập bảng so sánh |
| **6** | Chạy lại bước 2 trên model tối ưu → 3.2 render report cuối → P2 ablation nếu GPU rảnh → review → commit |

> Nếu cần rút ngắn: giảm `--trials` từ 40 xuống 20 mỗi nhánh (vẫn 80 trial/nhánh, đủ tin cậy với random search).

---

## Definition of Done chung

- [ ] Accuracy mặt ≥ 0,65 trên test 10 lớp
- [ ] Accuracy tiếng ≥ 0,70 trên test 10 lớp
- [ ] Accuracy fusion ≥ 0,70 và hơn nhánh đơn tốt nhất ≥ 3 điểm
- [ ] F1 macro ≥ 0,60
- [ ] Đã tune: mỗi nhánh có ≥ 4 model variants × ≥ 20 trial, có bảng top-10 trong docs
- [ ] Model cuối dùng cấu hình tối ưu từ tuning (không phải config mặc định)
- [ ] Có bảng so sánh baseline vs tuned trong `docs/progress_status.md`
- [ ] Latency end-to-end ≤ 200 ms/mẫu trên GPU
- [ ] `report.html` mở offline được, < 10 MB, đủ 6 nhóm nội dung bắt buộc
- [ ] Demo 1 lệnh chạy được, JSON đủ schema 10 lớp
- [ ] `pytest tests/unit` pass hết
- [ ] Không còn output cũ 7 lớp bị trích dẫn
- [ ] Tất cả docs khớp code, `docs/progress_status.md` là single source of truth
- [ ] Đã commit (Conventional Commits)
