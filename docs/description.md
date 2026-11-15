# Mô tả project Emotion-Fusion

## 1. Tên + tagline
- **Tên đầy đủ**: Nghiên cứu và xây dựng hệ thống nhận diện cảm xúc con người dựa trên biểu cảm khuôn mặt và tín hiệu âm thanh sử dụng Deep Learning
- **Slug**: `emotion-fusion`
- **Tagline**: Nhận diện 10 cảm xúc từ ảnh khuôn mặt và giọng nói bằng Deep Learning với fusion muộn; cả 2 nhánh dùng chung thang 10 lớp, báo cáo bằng một file HTML offline.

## 2. Problem — Bài toán là gì, vì sao quan trọng?
- **Bài toán**: Phân loại (classification) cảm xúc từ 2 modalities. Cả 2 nhánh dùng thang 10 lớp: angry (0), disgust (1), fear (2), happy (3), sad (4), surprise (5), neutral (6), contempt (7), boredom (8), excited (9). Nhánh mặt: 7 lớp gốc theo FER2013 + 3 lớp mới từ nguồn ảnh ngoài. Nhánh tiếng: 7 lớp gốc từ RAVDESS/CREMA-D + 3 lớp mới từ ASVP-ESD. Fusion gộp trực tiếp 10 lớp.
- **Vì sao quan trọng**: Cảm xúc là tín hiệu cốt lõi trong tương tác người–máy. Hệ thống hỗ trợ giáo dục (đánh giá hứng thú học tập), tư vấn tâm lý sơ bộ và chăm sóc khách hàng. Kết hợp mặt + tiếng giúp ổn định hơn chỉ dùng một kênh, đúng yêu cầu đa mô hình của đề tài. Việc mở rộng lên 10 lớp giải quyết điểm nghẽn thực tế: lớp `surprise` trước đây chỉ có 192 mẫu (2,2%), nay lên 1.778 mẫu (8,4%).
- **Ví dụ**: Cho ảnh mặt 48x48 đang cười + câu nói 4 giây vui vẻ → hệ thống trả về `happy` với confidence 0,87.

## 3. Target users — Ai dùng, ở đâu, kỳ vọng gì?
- **Người dùng chính**: Sinh viên thực hiện đồ án và giảng viên chấm bài; nhà nghiên cứu AI cần baseline đa mô hình gọn nhẹ.
- **Bối cảnh**: Chạy trên máy lab có GPU (fallback CPU-only), dữ liệu để sẵn trong `data/raw/`, xem kết quả qua `report.html` mở bằng double-click và `results/eda/report.html`, không cần mạng.
- **Kỳ vọng output**: Nhãn cảm xúc + phân bố xác suất (mặt 10 lớp, tiếng 10 lớp, fusion 10 lớp) + thời gian suy luận (ms). Kết quả trung thực, có cả case sai để phân tích.

## 4. Input / Output
- **Input nhánh mặt**: Ảnh xám 48x48, 1 kênh. Nguồn FER2013 (`data/raw/fer2013.csv`, cột emotion + pixels 2.304 số, 35.887 mẫu: train 28.709 / val 3.589 / test 3.589 theo cột `Usage` gốc) **+ 3.904 ảnh extra** cho 3 lớp mới (contempt 2.946 từ AffectNet, boredom 358 từ Student Engagement, excited 600 proxy từ AffectNet happy), tổng **39.791 mẫu 10 lớp**: train 31.441 / val 4.175 / test 4.175. Ví dụ `data/processed/demo/face_01.png`.
- **Input nhánh tiếng**: File WAV 16 kHz mono dài 3–5 giây, chuẩn hóa về 4 giây, chuyển thành Mel-spectrogram 128 mel x 128 khung. Nguồn gộp 3 bộ tổng 21.109 WAV, 246 speaker: RAVDESS 1.440 file (24 speaker) + CREMA-D 7.442 file (91 speaker) + ASVP-ESD bản Kaggle 12.227 file dùng được (131 speaker Audio + Bonus). Split theo speaker, seed 42: train 14.457 (172 speaker) / val 3.200 (36 speaker) / test 3.452 (38 speaker), giao nhau rỗng. Ví dụ `data/processed/demo/voice_01.wav`.
- **Output**: JSON gồm `emotion`, `confidence`, `probs_face[10]`, `probs_audio[10]`, `probs_fused[10]`, `inference_time_ms`. Ví dụ:
```json
{
  "emotion": "happy",
  "confidence": 0.87,
  "probs_fused": [0.02, 0.01, 0.01, 0.87, 0.03, 0.05, 0.01, 0.00, 0.00, 0.00],
  "inference_time_ms": 42
}
```
- **Splits**: 70/15/15, seed 42. Mặt giữ cột `Usage` gốc (stratify ngầm định) + ảnh extra thêm vào (`Usage=extra`, chia 70/15/15 riêng rồi gộp), tiếng split theo speaker để chống leakage. File split nằm ở `data/splits/face_*.csv` và `data/splits/audio_*.csv` kèm `audio_*_proc.csv` trỏ tới mel `.npy` đã vật liệu hoá đủ 21.109 file.

## 5. Core features
| Độ ưu tiên | Tính năng | Lý do |
|------------|-----------|-------|
| Must | Data pipeline mặt (48x48, 39.791 mẫu) + tiếng (mel 128x128), split đúng chuẩn | Không có dữ liệu sạch thì mọi model đều sai |
| Must | Face-CNN 10 lớp + Audio-CNN 10 lớp + late fusion 10 lớp | Yêu cầu cốt lõi của đề tài đa mô hình, đã khớp dữ liệu thực tế |
| Must | Eval 2 lớp + `report.html` offline | Chứng minh metric và cho giảng viên chấm không cần mạng |
| Must | Demo 1 lệnh `python -m src.demo` | Tiêu chí chạy được end-to-end |
| Should | Confusion matrix per-class (mặt 10x10, tiếng 10x10) + 5 case tệ nhất | Phân tích lỗi trung thực, tăng điểm báo cáo |
| Should | Thử realtime webcam/micro | Điểm cộng trình diễn nếu latency cho phép |
| Could | Export ONNX, Gradio UI, API FastAPI | Để tương lai, không làm trong MVP |

## 6. Out of scope (MVP không làm)
- Nhận diện nhiều người trong cùng một ảnh.
- Phân tích video dài hoặc hội thoại nhiều lượt.
- 4 nhãn ASVP-ESD còn lại: pleasure (10), pain (11), disappointment (12), breath (13) — đã loại ở bước `make_splits`.
- Triển khai mobile hoặc edge thiết bị yếu.

## 7. Success criteria
- Accuracy mặt ≥ 0,65 trên test 10 lớp (chấp nhận khó hơn vì 3 lớp mới ít mẫu: boredom 358, excited 600).
- Accuracy tiếng ≥ 0,70 trên test 10 lớp (chấp nhận khó hơn vì 3 lớp mới ít mẫu).
- Accuracy fusion ≥ 0,70 trên 10 lớp và cao hơn nhánh đơn tốt nhất ≥ 3 điểm, F1 macro ≥ 0,60.
- Latency end-to-end ≤ 200 ms/mẫu trên GPU lab.
- `report.html` dưới 10 MB, mở offline bằng double-click, số khớp `metrics.json`.
- Demo 1 lệnh chạy được trên máy sạch sau khi cài `requirements.txt`.
- Không leakage speaker: giao train/val/test rỗng (đã kiểm chứng).

## 8. Mô tả các module
### Module: data
- Làm gì: Load, augment và chia split cho 2 nhánh mặt và tiếng.
- Input: `fer2013.csv` + ảnh extra (`affectnet_face/contempt`, `face_extra/boredom`, `face_extra/excited`) + `audio/*.wav` + `crema/AudioWAV/*.wav` + `asvp/{Audio,Bonus}/actor_*/*.wav`.
- Output: `{'image': [1,48,48], 'label 0-9}` và `{'mel': [1,128,128], 'label 0-9}`.
- Tiêu chí thành công: 1 batch đúng shape, không leakage speaker, val/test không dính augment train, đủ 21.109 mel `.npy`.
- Ngoài phạm vi: Không train model trong module này.

### Module: models
- Làm gì: Face-CNN (ResNet-18 grayscale, 10 lớp) + Audio-CNN (4 khối conv + GAP, 10 lớp) + `fuse_probs` gộp 10 lớp.
- Input: Batch `[B,1,48,48]` hoặc `[B,1,128,128]`.
- Output: `probs_face [B,10]`, `probs_audio [B,10]`, `probs_fused [B,10]` sau fusion.
- Tiêu chí thành công: Forward 1 batch không lỗi shape, tổng params vừa VRAM lab.
- Ngoài phạm vi: Không viết trainer trong module này.

### Module: training
- Làm gì: Train riêng từng nhánh, lưu best theo F1 val, hỗ trợ resume và early stopping.
- Input: Loader train/val + `train_config.yaml`.
- Output: Checkpoint best trong `models/checkpoints/` + log `results/runs/<ts>/`.
- Tiêu chí thành công: Overfit được 1 batch nhỏ, loss val giảm ổn định, có best checkpoint.
- Ngoài phạm vi: Không tune trọng số fusion ở đây (để cho eval).

### Module: evaluation + visualization
- Làm gì: Tính accuracy, F1 macro, confusion matrix (mặt 10x10, tiếng 10x10, fusion 10x10), latency rồi render `report.html` offline.
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
