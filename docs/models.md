# Model Card — Emotion Fusion (Face-CNN + Audio-CNN + Late Fusion)

> Tài liệu kiến trúc model duy nhất. Đồng bộ với `configs/model_config.yaml` và `src/models/`.

## 1. Tổng quan
Hệ thống gồm 2 nhánh độc lập, mỗi nhánh ra phân bố xác suất 7 cảm xúc, sau đó gộp bằng late fusion có trọng số:

```
[Ảnh xám 48x48] → Face-CNN (ResNet-18 grayscale) → probs_face[7]
[WAV 16 kHz mono 4s] → Mel 128x128 → Audio-CNN (4 khối conv + GAP) → probs_audio[7]
probs_face + probs_audio → Late Fusion (w_face * p_face + w_audio * p_audio) → emotion + confidence
```

Mã nguồn: `src/models/face_cnn.py`, `src/models/audio_cnn.py`, `src/models/fusion.py`, dựng chung qua `src/models/architecture.py::build_model(kind)`.

## 2. Danh mục cảm xúc
Thứ tự cố định 7 lớp, dùng chung cho mặt và tiếng:

| ID | Nhãn | Ghi chú |
|----|------|---------|
| 0 | angry | Giận dữ |
| 1 | disgust | Ghê tởm (lớp hiếm nhất FER2013) |
| 2 | fear | Sợ hãi |
| 3 | happy | Vui vẻ |
| 4 | sad | Buồn |
| 5 | surprise | Ngạc nhiên |
| 6 | neutral | Trung tính |

Hằng số: `EMOTIONS` trong `src/models/fusion.py`.

## 3. Face-CNN — nhánh khuôn mặt
- **Backbone**: ResNet-18, `pretrained=True` theo `configs/model_config.yaml`.
- **Thích ứng ảnh xám**: `conv1` từ 3 kênh về 1 kênh, khởi tạo bằng trung bình trọng số RGB (`old.weight.mean(dim=1)`), giữ lại khả năng transfer learning. Xem `src/models/face_cnn.py`.
- **Head**: `fc` thay bằng `Linear(in_features, 7)`.
- **Input**: batch `[B,1,48,48]`, normalize mean 0,5 std 0,5.
- **Output**: logits `[B,7]`, softmax thành `probs[7]`.
- **Tham số**: khoảng 11,17M (ResNet-18 chuẩn).
- **Loss**: `CrossEntropyLoss` có trọng số lớp khi FER2013 mất cân bằng, dựng qua `build_loss(class_weights)`.

## 4. Audio-CNN — nhánh giọng nói
- **Kiến trúc**: 4 khối `Conv2d-BatchNorm-ReLU x2 + MaxPool2d(2)`, kênh tăng dần 32 → 64 → 128 → 256, sau đó `AdaptiveAvgPool2d(1)`. Xem `src/models/audio_cnn.py`.
- **Head**: `Flatten → Dropout(0,3) → Linear(256,128) → ReLU → Dropout(0,3) → Linear(128,7)`.
- **Input**: Mel-spectrogram log-mel `[B,1,128,128]` (128 mel x 128 khung, chuẩn hóa mean/std theo mẫu).
- **Output**: logits `[B,7]`, softmax thành `probs[7]`.
- **Tham số**: khoảng 1,21M, nhẹ hơn nhánh mặt nhiều, train nhanh trên CPU.
- **Tiền xử lý**: resample 16 kHz mono, cắt/pad 4 giây, `n_mels=128`, `power_to_db`, pad/cắt trục thời gian về 128. Xem `src/demo/__main__.py::load_mel_tensor` và `src/data/audio_dataset.py`.

## 5. Late Fusion
- **Công thức**: `p_fused = w_face * p_face + w_audio * p_audio`, sau đó chuẩn hóa tổng bằng 1. Mặc định `w_face=0,5`, `w_audio=0,5`.
- **Hàm**: `fuse_probs(p_face, p_audio, w_face, w_audio)` và `predict_fused(face_logits, audio_logits, ...)` trong `src/models/fusion.py`.
- **Tune trọng số**: `tune_on_val: true` trong `model_config.yaml`, học trên tập val ở giai đoạn eval, không tune trong lúc train nhánh.
- **Vì sao late fusion**: 2 modalities khác tốc độ học, train riêng ổn định hơn early fusion; dễ ablation mặt đơn / tiếng đơn / fusion; thêm nhánh mới chỉ cần thêm trọng số.

## 6. Config kiến trúc
File `configs/model_config.yaml`:

```yaml
face:
  name: face_resnet18_gray
  in_channels: 1
  num_classes: 7
  pretrained: true
audio:
  name: audio_cnn_4blocks
  in_channels: 1
  num_classes: 7
  dropout: 0.3
fusion:
  method: weighted_average
  w_face: 0.5
  w_audio: 0.5
  tune_on_val: true
```

Không hardcode số lớp, dropout hay pretrained trong code, mọi thứ đọc từ config qua `build_model(kind)`.

## 7. Kiểm tra nhanh
```bash
pytest tests/unit -q
python -m src.models --help  # nếu có CLI kiểm tra forward
```

Forward 1 batch phải đúng shape `[B,7]` cho cả 2 nhánh, fusion ra `[B,7]` chuẩn hóa tổng bằng 1.

## 8. Giới hạn và hướng nâng cấp
- Chỉ ảnh xám 48x48, chưa chịu mặt nghiêng mạnh hay che khuất.
- Chỉ WAV 16 kHz mono 3–5 giây, chưa chịu nhiễu nền lớn.
- Fusion tuyến tính đơn giản, chưa học attention theo mẫu.
- Hướng nâng cấp: SpecAugment mạnh hơn, học trọng số fusion trên val, thử backbone nhẹ (MobileNet) để chạy edge.
