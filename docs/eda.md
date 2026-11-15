# EDA — Phân tích khám phá dữ liệu Emotion-Fusion

> Chạy bằng `python notebooks/2026-11-03_eda.py`, plots lưu tại `results/eda/`. Dữ liệu chốt: FER2013 mặt 35.887 mẫu **+ 3.904 ảnh extra** (tổng 39.791 mẫu 10 lớp) + tiếng gộp 3 nguồn 21.109 WAV (1.440 RAVDESS + 7.442 CREMA-D + 12.227 ASVP-ESD, 246 speaker), seed 42.

## 1. Nhánh mặt (FER2013 + ảnh extra, ảnh xám 48x48)
FER2013 gốc: 35.887 mẫu, giữ cột `Usage` gốc: train 28.709 / val 3.589 / test 3.589. Sau khi gộp 3.904 ảnh extra (contempt 2.946, boredom 358, excited 600) → **tổng 39.791 mẫu**: train 31.441 / val 4.175 / test 4.175.

Bảng dưới đây là phân bố **FER2013 gốc** (chưa gồm ảnh extra), dùng để đánh giá mất cân bằng 7 lớp nền:

| Cảm xúc | Train | Val | Test | Nhận xét |
|---------|-------|-----|------|----------|
| angry (0) | 3.995 (13,9%) | 467 | 491 | Đủ mẫu |
| disgust (1) | 436 (1,5%) | 56 | 55 | Lớp hiếm nhất, cần weighted loss + augment |
| fear (2) | 4.097 (14,3%) | 496 | 528 | Đủ mẫu |
| happy (3) | 7.215 (25,1%) | 895 | 879 | Lớp đa số |
| sad (4) | 4.830 (16,8%) | 653 | 594 | Đủ mẫu |
| surprise (5) | 3.171 (11,0%) | 415 | 416 | Trung bình |
| neutral (6) | 4.965 (17,3%) | 607 | 626 | Đủ mẫu |

**3 lớp mới (sau khi gộp ảnh extra):**

| Cảm xúc | Train | Val | Test | Nguồn ảnh |
|---------|-------|-----|------|-----------|
| contempt (7) | 2.062 | 442 | 442 | AffectNet (crop 96x96) |
| boredom (8) | 250 | 54 | 54 | Student Engagement (center-crop) |
| excited (9) | 420 | 90 | 90 | **Proxy từ ảnh `happy` AffectNet** |

- Pixel mẫu (2.000 ảnh train): mean 129,6, std 64,6, min 0, max 255.
- Normalize đang dùng mean 0,5 std 0,5 sau khi đưa về [0,1] là hợp lý.
- Rủi ro: mất cân bằng disgust (gấp 16 lần so với happy) → bắt buộc `class weight` trong `build_loss` + báo F1 macro, không chỉ accuracy.
- Rủi ro mới: ảnh extra đến từ nguồn khác FER2013 (khác crop, khác chất lượng, khác phân bố nhân vật) → có thể ảnh hưởng accuracy 7 lớp gốc, cần ablation so sánh. Lớp `excited` là proxy từ `happy` nên không phải ground truth thật.

## 2. Nhánh tiếng (gộp RAVDESS + CREMA-D + ASVP-ESD, WAV 16 kHz mono)
Tổng 21.109 WAV (1.440 RAVDESS + 7.442 CREMA-D + 12.227 ASVP-ESD bản Kaggle, chỉ giữ 9/13 nhãn), 246 speaker, split theo speaker, seed 42: train 172 speaker (14.457) / val 36 speaker (3.200) / test 38 speaker (3.452). Không leakage speaker (giao nhau rỗng).

| Cảm xúc | Train | Val | Test | Tổng | Nhận xét |
|---------|-------|-----|------|------|----------|
| angry (0) | 1.949 | 443 | 524 | 2.916 | Gộp 3 nguồn, đủ mẫu |
| disgust (1) | 988 | 208 | 267 | 1.463 | RAVDESS + CREMA-D (ASVP mã 07 map sang contempt) |
| fear (2) | 1.778 | 431 | 434 | 2.643 | Gộp 3 nguồn |
| happy (3) | 2.063 | 570 | 698 | 3.331 | Gộp 3 nguồn |
| sad (4) | 3.083 | 620 | 454 | 4.157 | Lớp đa số nhánh tiếng |
| surprise (5) | 1.177 | 298 | 303 | 1.778 | Đã bù xong (trước chỉ 192) |
| neutral (6) | 2.056 | 446 | 471 | 2.973 | Gộp 3 nguồn |
| contempt (7) | 672 | 85 | 142 | 899 | Mới, chỉ ASVP-ESD |
| boredom (8) | 402 | 45 | 101 | 548 | Mới, chỉ ASVP-ESD |
| excited (9) | 289 | 54 | 58 | 401 | Mới, lớp nhỏ nhất nhánh tiếng |

- Nguồn train: 8.447 asvp + 4.990 crema + 1.020 ravdess; val: 1.916 asvp + 984 crema + 300 ravdess; test: 1.864 asvp + 1.468 crema + 120 ravdess.
- Mẫu kiểm tra: `Actor_01/03-01-01-01-01-01-01.wav` dài 3,30 giây, đúng chuẩn 3–5 giây, pad/cắt về 4 giây rồi mel 128x128.
- Rủi ro mới: 3 lớp mới contempt/boredom/excited chỉ có ở ASVP (NonSpeech nhiều) → bắt buộc class weight + SpecAugment + theo dõi F1 từng lớp; val chỉ 36 speaker nên vẫn theo dõi F1 macro thay vì accuracy.

## 3. Kết luận cho train
- Mặt: dùng weighted cross-entropy + augment Albumentations (flip, xoay ±10 độ), monitor `val_f1_macro`.
- Tiếng: giữ split speaker hiện tại, dùng SpecAugment time/freq mask, monitor per-class, đặc biệt 3 lớp mới contempt/boredom/excited.
- Cả hai: input đã chuẩn (mặt `[1,48,48]`, mel `[1,128,128]`), `data/processed/audio/{train,val,test}` đủ 21.109 file `.npy` + `audio_*_proc.csv`, sẵn sàng train sanity.

## 4. Plots và báo cáo HTML
- `results/eda/report.html` — báo cáo EDA offline một file duy nhất (khoảng 5,1 MB, mở bằng double-click, ảnh và âm thanh nhúng base64, không cần mạng). Sinh bằng `python notebooks/2026-11-03_eda_html.py`.
- Mục 3 trong báo cáo là gallery 10 cảm xúc: 7 lớp chung gồm ảnh mặt FER2013 48×48 + Mel + WAV; 3 lớp mới contempt/boredom/excited có Mel + WAV (ảnh mặt dùng ảnh giữ chỗ vì FER2013 không có).
- `results/eda/real_contempt/` — ảnh contempt thật từ AffectNet dùng cho nhánh mặt (7 file mẫu train), bổ sung gallery ảnh thật cho lớp mới.- File mẫu: `face_emotion_<id>_<label>.png`, `mel_emotion_<id>_<label>.png`, `audio_emotion_<id>_<label>.wav` trong `results/eda/`. Trích bằng `python notebooks/2026-11-03_eda_samples.py`.
- `results/eda/face_train_dist.png`, `face_val_dist.png`, `face_test_dist.png`
- `results/eda/audio_train_dist.png`, `audio_val_dist.png`, `audio_test_dist.png`
- `results/eda/face_sample.png`, `results/eda/audio_mel_sample.png`
