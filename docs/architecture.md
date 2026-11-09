# Kiến trúc hệ thống Emotion Fusion (late fusion)

## Tổng quan
Hai nhánh độc lập với số lớp khác nhau, gặp nhau ở 7 lớp chung khi fusion:

```
[Ảnh mặt 48x48 xám] → Face-CNN (ResNet-18 grayscale, 7 lớp) → probs_face[7]
[WAV 16kHz mono 4s] → Mel 128x128 → Audio-CNN (CNN 4 khối + GAP, 10 lớp) → probs_audio[10]
probs_face[7] + probs_audio[10] → _align_probs (cắt 7 lớp đầu, chuẩn hoá lại) → Late Fusion (w_face * p_face + w_audio * p_audio) → emotion + confidence
```

Mã nguồn: `src/models/face_cnn.py`, `src/models/audio_cnn.py`, `src/models/fusion.py` (`fuse_probs`, `predict_fused`).

## Module và trách nhiệm
- `src/data`: `face_dataset.py` (48x48, normalize mean 0,5 std 0,5), `audio_dataset.py` (mel 128x128, SpecAugment chỉ train, nhãn 0–9), `split.py` (70/15/15, seed 42, tiếng split theo speaker), `make_splits.py` (map RAVDESS 7 mã + CREMA-D 6 mã + ASVP-ESD 9/13 mã), `make_processed.py` (PNG + mel `.npy` + `audio_*_proc.csv`).
- `src/models`: `face_cnn.py` (`build_face_model`, 7 lớp), `audio_cnn.py` (`build_audio_model`, 10 lớp), `fusion.py` (`fuse_probs` căn 7 lớp chung).
- `src/training`: `trainer.py` (train riêng từng nhánh, lưu best theo F1 val, early stopping patience 7).
- `src/evaluation`: `metrics.py` (accuracy, F1 macro, confusion matrix 7x7 / 10x10 / 7x7 fusion), `visualize_html.py` (render `report.html`).
- `src/demo`: `__main__.py` (1 lệnh chạy ảnh + wav mẫu, đo latency).

## Số liệu đã chốt (kiểm chứng ngày 2026-11-09)
- Mặt: FER2013 35.887 mẫu (train 28.709 / val 3.589 / test 3.589 theo cột `Usage` gốc), 35.887 PNG trong `data/processed/face/`.
- Tiếng: 21.109 WAV, 246 speaker (RAVDESS 1.440 + CREMA-D 7.442 + ASVP-ESD 12.227), split train 14.457 (172 speaker) / val 3.200 (36) / test 3.452 (38), không leakage. Đủ 21.109 mel `.npy` trong `data/processed/audio/`.
- Nhãn tiếng 0–9: angry 2.916, disgust 1.463, fear 2.643, happy 3.331, sad 4.157, surprise 1.778, neutral 2.973, contempt 899, boredom 548, excited 401.

## Lý do late fusion có căn lớp
- Hai modalities khác tốc độ học, train riêng ổn định hơn early fusion.
- Nhánh tiếng bắt buộc 10 lớp để giữ contempt/boredom/excited, nhánh mặt chỉ 7 lớp theo FER2013; fusion cắt 7 lớp đầu rồi chuẩn hoá lại nên không vỡ shape.
- Dễ ablation: so sánh mặt đơn, tiếng đơn và fusion trên cùng 7 lớp chung và cùng metric.
- Dễ mở rộng: thêm nhánh mới chỉ cần thêm trọng số, không sửa backbone.
