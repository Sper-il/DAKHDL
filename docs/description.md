# Mô tả project Emotion-Fusion

## 1. Tên + tagline
- **Tên đầy đủ**: Hệ thống nhận diện cảm xúc con người dựa trên biểu cảm khuôn mặt và tín hiệu âm thanh sử dụng Deep Learning
- **Slug**: `emotion-fusion`
- **Tagline**: Nhận diện 7 cảm xúc cơ bản từ ảnh khuôn mặt và giọng nói bằng Deep Learning với fusion muộn, báo cáo bằng một file HTML offline.

## 2. Problem — Bài toán là gì, vì sao quan trọng?
- **Bài toán**: Phân loại (classification) 7 cảm xúc cơ bản: angry, disgust, fear, happy, sad, surprise, neutral từ 2 modalities: ảnh khuôn mặt và đoạn âm thanh giọng nói.
- **Vì sao quan trọng**: Cảm xúc là tín hiệu cốt lõi trong tương tác người–máy. Hệ thống hỗ trợ giáo dục (đánh giá hứng thú học tập), tư vấn tâm lý sơ bộ và chăm sóc khách hàng. Kết hợp mặt + tiếng giúp ổn định hơn chỉ dùng một kênh, đúng yêu cầu đa mô hình của đề tài.
- **Ví dụ**: Cho ảnh mặt 48x48 đang cười + câu nói 4 giây vui vẻ → hệ thống trả về `happy` với confidence 0,87.

## 3. Target users — Ai dùng, ở đâu, kỳ vọng gì?
- **Người dùng chính**: Sinh viên thực hiện đồ án và giảng viên chấm bài.
- **Bối cảnh**: Chạy trên máy lab có GPU (fallback CPU-only), dữ liệu để sẵn trong `data/raw/`, xem kết quả qua `report.html` mở bằng double-click, không cần mạng.
- **Kỳ vọng output**: Nhãn cảm xúc + phân bố xác suất 7 lớp cho từng nhánh và sau fusion + thời gian suy luận (ms). Kết quả trung thực, có cả case sai để phân tích.

## 4. Input / Output
- **Input nhánh mặt**: Ảnh xám 48x48, 1 kênh. Nguồn FER2013 (`data/raw/fer2013.csv`, cột emotion + pixels 2304 số). Ví dụ `data/processed/demo/face_01.png`.
- **Input nhánh tiếng**: File WAV 16 kHz mono dài 3–5 giây, chuẩn hóa về 4 giây, chuyển thành Mel-spectrogram 128 mel x 128 khung. Nguồn RAVDESS/CREMA-D (`data/raw/audio/*.wav`). Ví dụ `data/processed/demo/voice_01.wav`.
- **Output**: JSON gồm `emotion`, `confidence`, `probs_face[7]`, `probs_audio[7]`, `probs_fused[7]`, `inference_time_ms`. Ví dụ:
```json
{
  "emotion": "happy",
  "confidence": 0.87,
  "probs_fused": [0.02, 0.01, 0.01, 0.87, 0.03, 0.05, 0.01],
  "inference_time_ms": 42
}
```
- **Splits**: 70/15/15, seed 42. Mặt stratify theo label, tiếng split theo speaker để chống leakage. File split nằm ở `data/splits/face_*.csv` và `audio_*.csv`.

## 5. Core features
| Độ ưu tiên | Tính năng | Lý do |
|------------|-----------|-------|
| Must | Data pipeline mặt (48x48) + tiếng (mel 128x128), split đúng chuẩn | Không có dữ liệu sạch thì mọi model đều sai |
| Must | Face-CNN + Audio-CNN + late fusion có trọng số | Yêu cầu cốt lõi của đề tài đa mô hình |
| Must | Eval 2 lớp + `report.html` offline | Chứng minh metric và cho giảng viên chấm không cần mạng |
| Must | Demo 1 lệnh `python -m src.demo` | Tiêu chí chạy được end-to-end |
| Should | Confusion matrix per-class + 5 case tệ nhất | Phân tích lỗi trung thực, tăng điểm báo cáo |
| Should | Thử realtime webcam/micro | Điểm cộng trình diễn nếu latency cho phép |
| Could | Export ONNX, Gradio UI, API FastAPI | Để tương lai, không làm trong MVP |

## 6. Out of scope (MVP không làm)
- Nhận diện nhiều người trong cùng một ảnh.
- Phân tích video dài hoặc hội thoại nhiều lượt.
- Cảm xúc phức tạp ngoài 7 lớp cơ bản (contempt, stress, sarcasm).
- Triển khai mobile hoặc edge thiết bị yếu.

## 7. Success criteria
- Accuracy mặt ≥ 0,65 trên FER2013 test, Accuracy tiếng ≥ 0,70 trên RAVDESS test.
- Accuracy fusion ≥ 0,70 và cao hơn nhánh đơn tốt nhất ≥ 3 điểm, F1 macro ≥ 0,60.
- Latency end-to-end ≤ 200 ms/mẫu trên GPU lab.
- `report.html` dưới 10 MB, mở offline bằng double-click, số khớp `metrics.json`.
- Demo 1 lệnh chạy được trên máy sạch sau khi cài `requirements.txt`.

## 8. Mô tả các module
### Module: data
- Làm gì: Load, augment và chia split cho 2 nhánh mặt và tiếng.
- Input: `fer2013.csv` + `audio/*.wav`.
- Output: `{'image': [1,48,48], 'label'}` và `{'mel': [1,128,128], 'label'}`.
- Tiêu chí thành công: 1 batch đúng shape, không leakage speaker, val/test không dính augment train.
- Ngoài phạm vi: Không train model trong module này.

### Module: models
- Làm gì: Face-CNN (ResNet-18 grayscale) + Audio-CNN (4 khối conv + GAP) + `fuse_probs`.
- Input: Batch `[B,1,48,48]` hoặc `[B,1,128,128]`.
- Output: `probs [B,7]` mỗi nhánh, `probs_fused [B,7]` sau fusion.
- Tiêu chí thành công: Forward 1 batch không lỗi shape, tổng params vừa VRAM lab.
- Ngoài phạm vi: Không viết trainer trong module này.

### Module: training
- Làm gì: Train riêng từng nhánh, lưu best theo F1 val, hỗ trợ resume và early stopping.
- Input: Loader train/val + `train_config.yaml`.
- Output: Checkpoint best trong `models/checkpoints/` + log `results/runs/<ts>/`.
- Tiêu chí thành công: Overfit được 1 batch nhỏ, loss val giảm ổn định, có best checkpoint.
- Ngoài phạm vi: Không tune trọng số fusion ở đây (để cho eval).

### Module: evaluation + visualization
- Làm gì: Tính accuracy, F1 macro, confusion matrix, latency rồi render `report.html` offline.
- Input: Checkpoint best + test split + `eval_config.yaml`.
- Output: `metrics.json`, `plots/*.png`, `report.html` duy nhất.
- Tiêu chí thành công: Số trong HTML khớp JSON, đủ bảng pass/fail, đồ thị loss, mẫu tốt/xấu.
- Ngoài phạm vi: Không sửa model hay data trong module này.

### Module: demo
- Làm gì: Chạy 1 lệnh với ảnh + wav mẫu, đo latency live, sinh `index.html` rút gọn.
- Input: PNG + WAV trong `data/processed/demo/`.
- Output: JSON emotion + `results/demo/<ts>/index.html`.
- Tiêu chí thành công: Fresh clone + 1 lệnh là chạy được, output đúng format API đã chốt.
- Ngoài phạm vi: Không làm realtime webcam trong MVP.
