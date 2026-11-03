# Hệ thống nhận diện cảm xúc từ khuôn mặt và âm thanh (Emotion Fusion)

> Nhận diện 7 cảm xúc cơ bản từ ảnh khuôn mặt và giọng nói bằng Deep Learning với fusion muộn, báo cáo bằng một file HTML offline.

## 1. Project Overview
Project nghiên cứu và xây dựng hệ thống nhận diện cảm xúc đa mô hình cho đồ án môn học. Hệ thống gồm hai nhánh độc lập: nhánh mặt dùng CNN trên ảnh xám 48x48 (FER2013) và nhánh tiếng dùng CNN trên Mel-spectrogram (RAVDESS/CREMA-D), sau đó gộp xác suất bằng late fusion.

Đối tượng dùng là sinh viên và giảng viên: chạy trên GPU lab, xem kết quả qua `report.html` mở bằng double-click, demo bằng một lệnh duy nhất. Mọi quyết định Define và Design đã chốt trong `docs/progress_status.md`.

## 2. Problem Statement
Nhận diện 7 cảm xúc (angry, disgust, fear, happy, sad, surprise, neutral) từ khuôn mặt và giọng nói. Bài toán khó vì FER2013 mất cân bằng lớp, ảnh nhỏ 48x48, âm thanh nhiễu nền và khác speaker. Mục tiêu MVP: mỗi nhánh đạt Accuracy ≥ 0,65–0,70, fusion cải thiện ≥ 3 điểm so với nhánh đơn tốt nhất, latency end-to-end ≤ 200 ms/mẫu trên GPU.

## 3. Main Features
| Độ ưu tiên | Tính năng | Lý do |
|------------|-----------|-------|
| Must | Data pipeline mặt (48x48) + tiếng (mel 128x128), split 70/15/15 seed 42 | Không có dữ liệu sạch thì mọi model đều sai |
| Must | Face-CNN + Audio-CNN + late fusion có trọng số | Yêu cầu cốt lõi của đề tài đa mô hình |
| Must | Eval 2 lớp + `report.html` offline | Chứng minh metric và cho giảng viên chấm không cần mạng |
| Must | Demo 1 lệnh `python -m src.demo` | Tiêu chí chạy được end-to-end |
| Should | Confusion matrix per-class + 5 case tệ nhất | Phân tích lỗi trung thực, tăng điểm báo cáo |
| Should | Thử realtime webcam/micro | Điểm cộng trình diễn nếu latency cho phép |
| Could | Export ONNX, Gradio UI, API FastAPI | Để tương lai, không làm trong MVP |

Out of scope MVP: nhiều người cùng lúc, video dài, cảm xúc phức tạp ngoài 7 lớp cơ bản.

## 4. System Architecture
```
[Ảnh mặt 48x48] → [Face-CNN] → probs_face[7] ─┐
                                              ├→ [Late Fusion] → emotion + confidence
[WAV 16kHz] → [Mel 128x128] → [Audio-CNN] → probs_audio[7] ─┘
```

Chi tiết xem `docs/architecture.md`, luồng dữ liệu xem `docs/data-flow.md`, API từng module xem `docs/api.md`.

## 5. Folder Structure
```
emotion-fusion/
├── data/{raw, processed, splits, processed/demo/}  # dữ liệu (gitignore trừ demo mẫu)
├── src/{data, models, training, evaluation, demo/}  # source code
├── models/{checkpoints, trained, configs/}          # model đã train
├── notebooks/      # EDA
├── tests/         # unit + integration
├── docs/          # mục lục xem docs/README.md: architecture, data-flow, api, description, data, models, training, evaluation, visualization, demo, setup, runbook, decisions, progress_status, AI_Usage, CHANGELOG
├── configs/       # data/model/train/eval yaml
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

# 6. Evaluate + sinh report.html
python -m src.evaluation --config configs/eval_config.yaml
# Mở results/runs/<ts>/report.html bằng double-click

# 7. Demo 1 lệnh
python -m src.demo --image data/processed/demo/face_01.png --audio data/processed/demo/voice_01.wav

# 8. Test
pytest -q
```

## 7. Example Input → Output
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
  "probs_fused": [0.02, 0.01, 0.01, 0.87, 0.03, 0.05, 0.01],
  "inference_time_ms": 42
}
```

## 8. Results
Lần chạy mới nhất: `results/runs/<ts>/metrics.json` + báo cáo HTML `results/runs/<ts>/report.html` (mở bằng double-click, xem `modules/visualization.md`).

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
