# Runbook — Sổ vận hành End-to-End Emotion Fusion

> Mọi lệnh chạy theo đúng thứ tự: setup → split → sanity → train → eval → visualize → demo. Mỗi lệnh đều ghi rõ input, output và cách kiểm tra.

## 0. Quy ước đường dẫn
- Dữ liệu gốc: `data/raw/fer2013.csv`, `data/raw/audio/*.wav`.
- Split: `data/splits/face_*.csv`, `data/splits/audio_*.csv`.
- Checkpoint: `models/checkpoints/<face|audio>/best.pt`.
- Kết quả: `results/runs/<ts>/`, demo: `results/demo/<ts>/`.

## 1. Split dữ liệu (chạy 1 lần)
```bash
python -m src.data.make_splits
```
- **Kiểm tra**: đủ 6 file split, mặt train 28.709 / val 3.589 / test 3.589, tiếng train 960 / val 180 / test 300.
- **Verify loader**:
```bash
python -m src.data --config configs/data_config.yaml \
  --face-csv data/splits/face_val.csv \
  --audio-csv data/splits/audio_val.csv
```
- **Đạt**: batch mặt `[4,1,48,48]`, batch tiếng `[4,1,128,128]`.

## 2. Sanity check (2 epoch, bắt buộc trước train full)
```bash
python -m src.training --branch face \
  --train-csv data/splits/face_train.csv \
  --val-csv data/splits/face_val.csv --sanity

python -m src.training --branch audio \
  --train-csv data/splits/audio_train.csv \
  --val-csv data/splits/audio_val.csv --sanity
```
- **Đạt**: loss giảm, sinh `results/runs/<ts>_<branch>/config.yaml` và `models/checkpoints/<branch>/best.pt`.

## 3. Train full
```bash
python -m src.training --branch face \
  --train-csv data/splits/face_train.csv \
  --val-csv data/splits/face_val.csv

python -m src.training --branch audio \
  --train-csv data/splits/audio_train.csv \
  --val-csv data/splits/audio_val.csv
```
- **Theo dõi**: log epoch (train loss, val loss/acc/F1), TensorBoard trong `run_dir/tb/`, early stopping `patience=7` theo `val_f1_macro`.
- **Output**: `best.pt`, `best.state.pt`, `last.pt` trong `models/checkpoints/<branch>/`.

## 4. Đánh giá + sinh report.html
```bash
python -m src.evaluation \
  --face-ckpt models/checkpoints/face/best.pt \
  --face-test-csv data/splits/face_test.csv \
  --audio-ckpt models/checkpoints/audio/best.pt \
  --audio-test-csv data/splits/audio_test.csv
```
- **Output**: `results/runs/<ts>/metrics.json`, `plots/confusion_*.png`, `report.html`.
- **Kiểm tra**: mở `report.html` bằng double-click trên máy không mạng, số khớp `metrics.json`, bảng pass/fail đúng target trong `configs/eval_config.yaml`.

## 5. Demo 1 lệnh
```bash
python -m src.demo \
  --image data/processed/demo/face_01.png \
  --audio data/processed/demo/voice_01.wav \
  --face-ckpt models/checkpoints/face/best.pt \
  --audio-ckpt models/checkpoints/audio/best.pt
```
- **Output**: `results/demo/<ts>/result.json` + `index.html`.
- **Đạt**: JSON đủ `{emotion, confidence, probs_fused[7], inference_time_ms}`, latency ≤ 200 ms trên GPU.

## 6. Test hồi quy
```bash
pytest tests/unit -q
```
- **Đạt**: 3/3 pass, không warning nghiêm trọng.

## 7. Khi train fail xem ở đâu
- Log train: console + `results/runs/<ts>_<branch>/config.yaml` + TensorBoard.
- Log eval: `results/runs/<ts>/metrics.json` + `report.html` mục Bad Cases.
- Lỗi data shape: chạy lại bước 1, đối chiếu `docs/data.md` và `docs/data-flow.md`.
- Lỗi model shape: đối chiếu `docs/models.md` (face `[B,1,48,48]`, audio `[B,1,128,128]`, output `[B,7]`).
