# Model Card — Emotion Fusion (Face-CNN + Audio-CNN + Late Fusion)

> Tài liệu kiến trúc model duy nhất. Đồng bộ với `configs/model_config.yaml` và `src/models/`.

## 1. Tổng quan
Hệ thống gồm 2 nhánh độc lập cùng thang 10 lớp, sau đó gộp bằng late fusion có trọng số:

```
[Ảnh xám 48x48] → Face-CNN (ResNet-18 grayscale) → probs_face[10]
[WAV 16 kHz mono 4s] → Mel 128x128 → Audio-CNN (4 khối conv + GAP) → probs_audio[10]
probs_face + probs_audio → Late Fusion (w_face * p_face + w_audio * p_audio) → emotion + confidence
```

Mã nguồn: `src/models/face_cnn.py`, `src/models/audio_cnn.py`, `src/models/fusion.py`, dựng chung qua `src/models/architecture.py::build_model(kind)`.

## 1b. Các biến thể model (hyperparameter tuning)

Để tìm model tối ưu, mỗi nhánh được thử **4 kiến trúc** × **6 siêu tham số** (mỗi siêu tham số 6–7 giá trị), random search 40 trial/variant = **160 trial mỗi nhánh**. Không gian tìm kiếm khai báo tại `configs/tuning_space.yaml`, engine tại `src/training/tuner.py`.

| Nhánh | Variant | Kiến trúc | Ghi chú |
|-------|---------|-----------|---------|
| Mặt | `resnet18_gray` | ResNet-18 1 kênh, pretrained | Baseline hiện tại |
| Mặt | `resnet18_gray_nopretrain` | ResNet-18 1 kênh, không pretrained | Kiểm chứng pretrained có thật sự giúp không |
| Mặt | `mobilenetv2_gray` | MobileNetV2 1 kênh, pretrained | Nhẹ hơn nhiều → hợp latency ≤ 200 ms |
| Mặt | `small_cnn` | CNN tự viết 4 block | Baseline tối giản, đối chứng |
| Tiếng | `cnn_4blocks` | 4 conv + GAP + MLP | Baseline hiện tại |
| Tiếng | `cnn_3blocks_wide` | 3 conv, kênh `[64,128,256]` | Ít layer, nhiều capacity hơn |
| Tiếng | `cnn_5blocks` | 5 conv, kênh `[16,32,64,128,256]` | Tăng độ sâu, tham khảo VGG-style |
| Tiếng | `cnn_4blocks_deep` | 4 conv + dropout cao | Kiểm chứng dropout ảnh hưởng thế nào |

**6 siêu tham số** mỗi nhánh: `learning_rate`, `weight_decay`, `batch_size`, `dropout`, `label_smoothing`, và `class_weight_power` (mặt) / `spec_augment_prob` (tiếng).

> **Quy tắc:** chỉ chọn siêu tham số trên **val**. Test set giữ kín tới bước eval cuối. Chi tiết ở `docs/plan_remaining.md` khối T (T0–T6).

## 2. Danh mục cảm xúc
Thứ tự cố định dùng chung cho **cả 2 nhánh**, 10 lớp:

| ID | Nhãn | Ghi chú |
|----|------|---------|
| 0 | angry | Giận dữ |
| 1 | disgust | Ghê tởm (lớp hiếm nhất FER2013) |
| 2 | fear | Sợ hãi |
| 3 | happy | Vui vẻ |
| 4 | sad | Buồn |
| 5 | surprise | Ngạc nhiên (CREMA-D không có, bù từ ASVP-ESD) |
| 6 | neutral | Trung tính (gộp calm) |
| 7 | contempt | Khinh bỉ — ảnh từ AffectNet, tiếng từ ASVP-ESD |
| 8 | boredom | Chán nản — ảnh từ Student Engagement, tiếng từ ASVP-ESD |
| 9 | excited | Hào hứng — ảnh là proxy từ AffectNet happy, tiếng từ ASVP-ESD |

Hằng số: `EMOTIONS` (10 lớp) và `EMOTIONS_7` (7 lớp gốc, dùng để đối chiếu) trong `src/models/fusion.py`. Cả `src/data/face_dataset.py` và `src/data/audio_dataset.py` dùng cùng thứ tự.

## 3. Face-CNN — nhánh khuôn mặt (10 lớp)
- **Backbone**: ResNet-18, `pretrained=True` theo `configs/model_config.yaml`.
- **Thích ứng ảnh xám**: `conv1` từ 3 kênh về 1 kênh, khởi tạo bằng trung bình trọng số RGB (`old.weight.mean(dim=1)`), giữ lại khả năng transfer learning. Xem `src/models/face_cnn.py`.
- **Head**: `fc` thay bằng `Linear(in_features, 10)`.
- **Input**: batch `[B,1,48,48]`, normalize mean 0,5 std 0,5.
- **Output**: logits `[B,10]`, softmax thành `probs[10]`.
- **Tham số**: khoảng 11,17M (ResNet-18 chuẩn).
- **Loss**: `CrossEntropyLoss` có trọng số lớp khi mất cân bằng (FER2013 disgust 436 mẫu, lớp mới boredom 358), dựng qua `build_loss(class_weights)`.

## 4. Audio-CNN — nhánh giọng nói (10 lớp)
- **Kiến trúc**: 4 khối `Conv2d-BatchNorm-ReLU x2 + MaxPool2d(2)`, kênh tăng dần 32 → 64 → 128 → 256, sau đó `AdaptiveAvgPool2d(1)`. Xem `src/models/audio_cnn.py`.
- **Head**: `Flatten → Dropout(0,3) → Linear(256,128) → ReLU → Dropout(0,3) → Linear(128,10)`.
- **Input**: Mel-spectrogram log-mel `[B,1,128,128]` (128 mel x 128 khung, chuẩn hóa mean/std theo mẫu).
- **Output**: logits `[B,10]` (7 lớp gốc + contempt/boredom/excited), softmax thành `probs[10]`.
- **Tham số**: khoảng 1,21M, nhẹ hơn nhánh mặt nhiều, train nhanh trên CPU.
- **Nguồn nhãn**: RAVDESS (7 lớp) + CREMA-D (6 lớp, thiếu surprise) + ASVP-ESD (bù surprise/contempt/boredom/excited). Xem `ASVP_MAP` trong `src/data/make_splits.py`.

## 5. Late Fusion
- **Công thức**: `p_fused = w_face * p_face + w_audio * p_audio`, sau đó chuẩn hóa tổng bằng 1. Mặc định `w_face=0,5`, `w_audio=0,5`.
- **Căn lớp**: cả 2 nhánh đều `[10]` cùng thứ tự label → `_align_probs` căn theo `min` nên giữ nguyên 10 lớp. Vẫn hỗ trợ fallback cắt về 7 lớp nếu 1 nhánh dùng checkpoint cũ.
- **Hàm**: `fuse_probs(p_face, p_audio, w_face, w_audio)` và `predict_fused(face_logits, audio_logits, ...)` trong `src/models/fusion.py`.
- **Tune trọng số**: `tune_on_val: true` trong `model_config.yaml`, học trên tập val ở giai đoạn eval, không tune trong lúc train nhánh.
- **Vì sao late fusion**: 2 modalities khác tốc độ học, train riêng ổn định hơn early fusion; dễ ablation mặt đơn / tiếng đơn / fusion; thêm nhánh mới chỉ cần thêm trọng số.

## 6. Config kiến trúc
File `configs/model_config.yaml`:

```yaml
face:
  name: face_resnet18_gray
  in_channels: 1
  num_classes: 10
  pretrained: true
audio:
  name: audio_cnn_4blocks
  in_channels: 1
  num_classes: 10
  dropout: 0.3
fusion:
  method: weighted_average
  w_face: 0.5
  w_audio: 0.5
  tune_on_val: true
```

Số lớp nằm trong config, không hardcode trong code. `build_model(kind)` đọc từ config; `demo/` và `evaluation/` tự suy `num_classes` từ shape checkpoint để tương thích checkpoint cũ.

## 7. Kiểm tra nhanh
```bash
pytest tests/unit -q
python -m src.models  # forward 1 batch giả, in shape + số tham số
```

Forward 1 batch phải đúng shape mặt `[B,10]`, tiếng `[B,10]`, fusion ra `[B,10]` chuẩn hóa tổng bằng 1. Test suite gồm 7 case: fusion 10 lớp, fusion tương thích ckpt 7 lớp, fusion cùng shape, forward `FaceCNN`, forward `AudioCNN`, shape SpecAugment, label range.

## 8. Giới hạn và hướng nâng cấp
- Chỉ ảnh xám 48x48, chưa chịu mặt nghiêng mạnh hay che khuất.
- Chỉ WAV 16 kHz mono 3–5 giây, chưa chịu nhiễu nền lớn.
- Ảnh lớp mới (contempt/boredom/excited) đến từ nguồn ngoài FER2013, khác domain; `excited` là proxy từ ảnh `happy`.
- Fusion tuyến tính đơn giản, chưa học attention theo mẫu.
- Hướng nâng cấp: SpecAugment mạnh hơn, học trọng số fusion trên val, thử backbone nhẹ (MobileNet) để chạy edge.
