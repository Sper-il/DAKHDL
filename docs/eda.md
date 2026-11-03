# EDA — Phân tích khám phá dữ liệu Emotion-Fusion

> Chạy bằng `python notebooks/2026-11-03_eda.py`, plots lưu tại `results/eda/`. Dữ liệu chốt: FER2013 mặt 35.887 mẫu + RAVDESS tiếng 1.440 WAV, seed 42.

## 1. Nhánh mặt (FER2013, ảnh xám 48x48)
Tổng 35.887 mẫu, giữ cột `Usage` gốc: train 28.709 / val 3.589 / test 3.589.

| Cảm xúc | Train | Val | Test | Nhận xét |
|---------|-------|-----|------|----------|
| angry (0) | 3.995 (13,9%) | 467 | 491 | Đủ mẫu |
| disgust (1) | 436 (1,5%) | 56 | 55 | Lớp hiếm nhất, cần weighted loss + augment |
| fear (2) | 4.097 (14,3%) | 496 | 528 | Đủ mẫu |
| happy (3) | 7.215 (25,1%) | 895 | 879 | Lớp đa số |
| sad (4) | 4.830 (16,8%) | 653 | 594 | Đủ mẫu |
| surprise (5) | 3.171 (11,0%) | 415 | 416 | Trung bình |
| neutral (6) | 4.965 (17,3%) | 607 | 626 | Đủ mẫu |

- Pixel mẫu (2.000 ảnh train): mean 129,6, std 64,6, min 0, max 255.
- Normalize đang dùng mean 0,5 std 0,5 sau khi đưa về [0,1] là hợp lý.
- Rủi ro: mất cân bằng disgust (gấp 16 lần so với happy) → bắt buộc `class weight` trong `build_loss` + báo F1 macro, không chỉ accuracy.

## 2. Nhánh tiếng (RAVDESS, WAV 16 kHz mono)
Tổng 1.440 WAV, 24 speaker, split theo speaker, seed 42: train 16 speaker (960) / val 3 speaker (180) / test 5 speaker (300). Không پاکستان leakage speaker (giao nhau rỗng).

| Cảm xúc | Train | Val | Test | Nhận xét |
|---------|-------|-----|------|----------|
| angry (0) | 128 | 24 | 40 | Cân bằng |
| disgust (1) | 128 | 24 | 40 | Cân bằng |
| fear (2) | 128 | 24 | 40 | Cân bằng |
| happy (3) | 128 | 24 | 40 | Cân bằng |
| sad (4) | 128 | 24 | 40 | Cân bằng |
| surprise (5) | 128 | 24 | 40 | Cân bằng |
| neutral (6) | 192 | 36 | 60 | Gồm neutral + calm nên gấp 1,5 lần |

- Mẫu kiểm tra: `Actor_03/03-01-01-01-01-01-03.wav` dài 3,44 giây, đúng chuẩn 3–5 giây, pad/cắt về 4 giây rồi mel 128x128.
- Train speakers: actor_03, 04, 05, 06, 07, 08, 11, 13, 15, 16, 18, 20, 21, 22, 23, 24.
- Val speakers: actor_12, 14, 19. Test speakers: actor_01, 02, 09, 10, 17.
- Rủi ro: neutral trội hơn → vẫn cần theo dõi F1 per-class; val chỉ 3 speaker nên phương sai cao, không tune quá kỹ trên val.

## 3. Kết luận cho train
- Mặt: dùng weighted cross-entropy + augment Albumentations (flip, xoay ±10 độ), monitor `val_f1_macro`.
- Tiếng: giữ split speaker hiện tại, dùng SpecAugment time/freq mask, monitor per-class neutral.
- Cả hai: input đã chuẩn (mặt `[1,48,48]`, mel `[1,128,128]`), sẵn sàng train sanity.

## 4. Plots và báo cáo HTML
- `results/eda/report.html` — báo cáo EDA offline một file duy nhất (312 KB, mở bằng double-click, ảnh nhúng base64, không cần mạng). Sinh bằng `python notebooks/2026-11-03_eda_html.py`.
- `results/eda/face_train_dist.png`, `face_val_dist.png`, `face_test_dist.png`
- `results/eda/audio_train_dist.png`, `audio_val_dist.png`, `audio_test_dist.png`
- `results/eda/face_sample.png`, `results/eda/audio_mel_sample.png`
