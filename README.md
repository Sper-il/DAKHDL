# Hệ thống nhận diện cảm xúc từ khuôn mặt và âm thanh (Emotion Fusion)

> Nhận diện 10 cảm xúc từ ảnh khuôn mặt và giọng nói bằng Deep Learning với fusion muộn; cả 2 nhánh dùng chung thang 10 lớp, báo cáo bằng một file HTML offline.

## 1. Project Overview
Project nghiên cứu và xây dựng hệ thống nhận diện cảm xúc đa mô hình cho đồ án môn học. Hệ thống gồm hai nhánh độc lập: nhánh mặt dùng CNN trên ảnh xám 48x48 (FER2013 + ảnh bổ sung, 39.791 mẫu, 10 lớp) và nhánh tiếng dùng CNN trên Mel-spectrogram (gộp RAVDESS + CREMA-D + ASVP-ESD, 21.109 WAV, 10 lớp), sau đó gộp xác suất bằng late fusion trên cùng thang 10 lớp.

Đối tượng dùng là sinh viên và giảng viên: chạy trên GPU lab, xem kết quả qua `report.html` mở bằng double-click, demo bằng một lệnh duy nhất. Mọi quyết định Define và Design đã chốt trong `docs/progress_status.md`.

> 📋 **Trạng thái hiện tại:** DATA ✅ · MODELS ✅ · TRAINING (code ✅, chưa train full) · EVALUATION ⏳ · VISUALIZE ⏳ · DEMO ⏳
> Plan công việc còn lại theo từng bước + phân công cho thành viên: **`docs/plan_remaining.md`**.

## 2. Problem Statement
Nhận diện cảm xúc từ khuôn mặt và giọng nói, cả 2 nhánh dùng thang 10 lớp: 7 lớp cơ bản (angry, disgust, fear, happy, sad, surprise, neutral) cộng 3 lớp mới (contempt, boredom, excited). Bài toán khó vì FER2013 mất cân bằng lớp, ảnh nhỏ 48x48, âm thanh nhiễu nền và khác speaker, 3 lớp mới ít mẫu (mặt boredom 358, excited 600; tiếng excited 401). Mục tiêu MVP: mặt ≥ 0,65, tiếng ≥ 0,70, fusion 10 lớp ≥ 0,70 và hơn nhánh đơn tốt nhất ≥ 3 điểm, latency end-to-end ≤ 200 ms/mẫu trên GPU.

## 3. Main Features
| Độ ưu tiên | Tính năng | Lý do |
|------------|-----------|-------|
| Must | Data pipeline mặt (48x48) + tiếng (mel 128x128), split 70/15/15 seed 42 | Không có dữ liệu sạch thì mọi model đều sai |
| Must | Face-CNN 10 lớp + Audio-CNN 10 lớp + late fusion 10 lớp | Yêu cầu cốt lõi của đề tài đa mô hình |
| Must | Eval 2 lớp + `report.html` offline — ⚠️ *chưa triển khai* | Chứng minh metric và cho giảng viên chấm không cần mạng |
| Must | Demo 1 lệnh `python -m src.demo` — ⚠️ *chưa triển khai* | Tiêu chí chạy được end-to-end |
| Should | Confusion matrix per-class + 5 case tệ nhất | Phân tích lỗi trung thực, tăng điểm báo cáo |
| Should | Thử realtime webcam/micro | Điểm cộng trình diễn nếu latency cho phép |
| Could | Export ONNX, Gradio UI, API FastAPI | Để tương lai, không làm trong MVP |

Out of scope MVP: nhiều người cùng lúc, video dài, 4 nhãn ASVP còn lại (pleasure/pain/disappointment/breath).

## 4. System Architecture
```
[Ảnh mặt 48x48] → [Face-CNN 10 lớp] → probs_face[10] ─┐
                                                      ├→ [Late Fusion 10 lớp] → emotion + confidence
[WAV 16kHz] → [Mel 128x128] → [Audio-CNN 10 lớp] → probs_audio[10] ─┘
```

Chi tiết xem `docs/architecture.md`, luồng dữ liệu xem `docs/data-flow.md`, API từng module xem `docs/api.md`.

## 5. Folder Structure
```
emotion-fusion/
├── data/{raw, processed, splits, processed/demo/}  # dữ liệu (gitignore trừ demo mẫu)
├── src/{data, models, training, evaluation, demo/}  # source code
├── models/{checkpoints, trained, configs/}          # model đã train
├── notebooks/      # EDA
├── tests/         # unit + integration (test_pending_modules.py kiểm tra module chưa làm luôn trống)
├── docs/          # mục lục xem docs/README.md: architecture, data-flow, api, description, data, models, training, evaluation, visualization, demo, setup, runbook, decisions, progress_status, plan_remaining, AI_Usage, CHANGELOG
├── configs/       # data/model/train/eval yaml
├── scripts/       # check_pending_modules.py — canh gác trạng thái module
└── results/{runs/<ts>/{metrics.json, report.html}, demo/<ts>/}
```

## 6. Installation / Run / Test
```bash
# 1. Clone + vào thư mục
cd emotion-fusion

# 2. Tạo môi trường
python -m venv .venv
.venv\Scripts\activate  # Windows

# 3. Cài dependency
pip install -r requirements.txt

# 4. Chuẩn bị data (tải FER2013 + RAVDESS theo docs/data-flow.md)
# Đặt fer2013.csv vào data/raw/, wav vào data/raw/audio/

# 5. Train từng nhánh rồi fusion (làm sau khi module data xong)
python -m src.training --config configs/train_config.yaml

# 6. Evaluate + sinh report.html  ⚠️ chưa triển khai
# python -m src.evaluation --config configs/eval_config.yaml
# Mở results/runs/<ts>/report.html bằng double-click

# 7. Demo 1 lệnh  ⚠️ chưa triển khai
# python -m src.demo --image data/processed/demo/face_01.png --audio data/processed/demo/voice_01.wav

# 8. Test
pytest -q
# Kiểm tra file của các bước chưa làm luôn trống
python scripts/check_pending_modules.py
```

## 7. Example Input → Output
> ⚠️ Schema dưới đây là **đặc tả thiết kế**, chưa có output thật — module `src/demo/` chưa triển khai.

**Input**
```json
{
  "image_path": "data/processed/demo/face_01.png",
  "audio_path": "data/processed/demo/voice_01.wav"
}
```

**Output**
```json
{
  "emotion": "happy",
  "confidence": 0.87,
  "probs_fused": [0.02, 0.01, 0.01, 0.87, 0.03, 0.05, 0.01, 0.00, 0.00, 0.00],
  "inference_time_ms": 42
}
```

Thứ tự lớp: `angry, disgust, fear, happy, sad, surprise, neutral, contempt, boredom, excited`.

## 8. Results
**Chưa có kết quả hợp lệ.** Module `src/evaluation/` đã bị xoá (để trống) ngày 2026-11-15 để xây lại từ đầu; checkpoint hiện tại mới chỉ train **sanity 1–2 epoch** (best_f1: mặt 0,019 / tiếng 0,303) nên chưa dùng được.

Run cũ `results/runs/20261109-1704/` chạy trên model **7 lớp** (trước khi mở rộng lên 10 lớp), không có nhánh fusion và `report.html` chỉ 2,7 KB → **không được trích dẫn trong báo cáo**.

Khi có kết quả mới, điền bảng sau:

| Metric | Target | Đạt được | Run |
|--------|--------|----------|-----|
| Accuracy mặt | ≥ 0,65 | TBD | — |
| Accuracy tiếng | ≥ 0,70 | TBD | — |
| Accuracy fusion | ≥ 0,70 và hơn nhánh đơn ≥ 3 điểm | TBD | — |
| Latency end-to-end | ≤ 200 ms | TBD | — |

## 9. Limitations
- Chỉ ảnh xám 48x48, chưa xử lý mặt nghiêng mạnh hoặc che khuất.
- Chỉ WAV 16 kHz mono 3–5 giây, chưa chịu nhiễu nền lớn.
- Chưa nhiều người cùng lúc, chưa video dài.

## 10. Future Improvements
- Thêm SpecAugment và augment mặt để tăng F1 lớp hiếm.
- Học trọng số fusion trên tập val thay vì cố định 0,5/0,5.
- Export ONNX và Gradio demo nếu cần trình diễn realtime.

---

Được duy trì với skill **ai-fusion-workflow**.
Xem `docs/AI_Usage.md` để biết cách AI được sử dụng trong project.
