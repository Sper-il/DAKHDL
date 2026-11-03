# Kiến trúc hệ thống Emotion Fusion (late fusion)

## Tổng quan
Hai nhánh độc lập, mỗi nhánh ra phân bố 7 cảm xúc, sau đó gộp có trọng số:

```
[Ảnh mặt 48x48 xám] → Face-CNN (baseline: ResNet-18 grayscale) → probs_face[7]
[WAV 16kHz mono] → Mel 128x128 → Audio-CNN (baseline: CNN 4 khối + GAP) → probs_audio[7]
probs_face + probs_audio → Late Fusion (w_face * p_face + w_audio * p_audio) → emotion + confidence
```

## Module và trách nhiệm
- `src/data`: `face_dataset.py` (48x48, normalize mean 0,5 std 0,5), `audio_dataset.py` (mel 128x128, SpecAugment chỉ train), `split.py` (70/15/15, seed 42, split audio theo speaker).
- `src/models`: `face_cnn.py` (`build_face_model`), `audio_cnn.py` (`build_audio_model`), `fusion.py` (`fuse_probs`).
- `src/training`: `trainer.py` (train riêng từng nhánh, lưu best theo F1 val).
- `src/evaluation`: `metrics.py` (accuracy, F1 macro, confusion matrix), `visualize_html.py` (render `report.html`).
- `src/demo`: `__main__.py` (1 lệnh chạy ảnh + wav mẫu, đo latency).

## Lý do late fusion
- Hai modalities khác tốc độ học, train riêng ổn định hơn early fusion.
- Dễ ablation: so sánh mặt đơn, tiếng đơn và fusion trên cùng metric.
- Dễ mở rộng: thêm nhánh mới chỉ cần thêm trọng số, không sửa backbone.
