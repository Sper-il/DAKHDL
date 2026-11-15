# Kiến trúc hệ thống Emotion Fusion (late fusion)

## Tổng quan
Hai nhánh độc lập cùng thang 10 lớp, gặp nhau khi fusion:

```
[Ảnh mặt 48x48 xám] → Face-CNN (ResNet-18 grayscale, 10 lớp) → probs_face[10]
[WAV 16kHz mono 4s] → Mel 128x128 → Audio-CNN (CNN 4 khối + GAP, 10 lớp) → probs_audio[10]
probs_face[10] + probs_audio[10] → _align_probs (căn về min số lớp, chuẩn hoá lại) → Late Fusion (w_face * p_face + w_audio * p_audio) → emotion + confidence
```

Mã nguồn: `src/models/face_cnn.py`, `src/models/audio_cnn.py`, `src/models/fusion.py` (`fuse_probs`, `predict_fused`).

## Module và trách nhiệm
- `src/data`: `face_dataset.py` (48x48, normalize mean 0,5 std 0,5, nhãn 0–9), `audio_dataset.py` (mel 128x128, SpecAugment chỉ train, nhãn 0–9), `split.py` (70/15/15, seed 42, tiếng split theo speaker), `make_splits.py` (map RAVDESS 7 mã + CREMA-D 6 mã + ASVP-ESD 9/13 mã), `add_face_extra.py` (gộp ảnh extra cho lớp 7–9 vào face splits), `make_processed.py` (PNG + mel `.npy` + `audio_*_proc.csv`).
- `src/models`: `face_cnn.py` (`build_face_model`, 10 lớp), `audio_cnn.py` (`build_audio_model`, 10 lớp), `fusion.py` (`fuse_probs` gộp 10 lớp).
- `src/training`: `trainer.py` (train riêng từng nhánh, lưu best theo F1 val, early stopping patience 7).
- `src/evaluation`: `metrics.py` (accuracy, F1 macro, confusion matrix 10x10 / 10x10 / 10x10 fusion), `visualize_html.py` (render `report.html`).
- `src/demo`: `__main__.py` (1 lệnh chạy ảnh + wav mẫu, đo latency, tự suy `num_classes` từ checkpoint).

## Số liệu đã chốt (kiểm chứng ngày 2026-11-15)
- Mặt: FER2013 35.887 mẫu (train 28.709 / val 3.589 / test 3.589 theo cột `Usage` gốc) **+ 3.904 ảnh extra** (contempt 2.946, boredom 358, excited 600) = **39.791 mẫu 10 lớp**; split cuối train 31.441 / val 4.175 / test 4.175.
- Tiếng: 21.109 WAV, 246 speaker (RAVDESS 1.440 + CREMA-D 7.442 + ASVP-ESD 12.227), split train 14.457 (172 speaker) / val 3.200 (36) / test 3.452 (38), không leakage. Đủ 21.109 mel `.npy` trong `data/processed/audio/`.
- Nhãn 0–9 (cả 2 nhánh cùng thứ tự): angry, disgust, fear, happy, sad, surprise, neutral, contempt, boredom, excited.
- Phân bố nhãn nhánh mặt (train / val / test): angry 3995/467/491, disgust 436/56/55, fear 4097/496/528, happy 7215/895/879, sad 4830/653/594, surprise 3171/415/416, neutral 4965/607/626, contempt 2062/442/442, boredom 250/54/54, excited 420/90/90.
- Phân bố nhãn nhánh tiếng (train / val / test): angry 1949/443/524, disgust 988/208/267, fear 1778/431/434, happy 2063/570/698, sad 3083/620/454, surprise 1177/298/303, neutral 2056/446/471, contempt 672/85/142, boredom 402/45/101, excited 289/54/58.

## Lý do late fusion có căn lớp
- Hai modalities khác tốc độ học, train riêng ổn định hơn early fusion.
- Cả 2 nhánh đều 10 lớp cùng thứ tự label → `_align_probs` căn theo `min` nên giữ nguyên 10 lớp, không mất thông tin lớp nào. Vẫn giữ fallback cắt về 7 lớp cho checkpoint cũ.
- Dễ ablation: so sánh mặt đơn, tiếng đơn và fusion trên cùng 10 lớp và cùng metric.
- Dễ mở rộng: thêm nhánh mới chỉ cần thêm trọng số, không sửa backbone.
