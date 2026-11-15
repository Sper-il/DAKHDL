# Dữ liệu Emotion Fusion

- `data/raw/` (gitignore): đặt `fer2013.csv` và thư mục `audio/*.wav` tại đây, không sửa trực đ��p.
- `data/raw/crema/AudioWAV/` (gitignore, khuyến nghị): thêm 7.442 file WAV CREMA-D (91 speaker) để tăng đa dạng giọng nói. Tải bằng `python -m src.data.download_cremad` hoặc thủ công từ [Kaggle ejlok1/cremad](https://www.kaggle.com/datasets/ejlok1/cremad).
- `data/raw/asvp/{Audio,Bonus}/actor_X/*.wav` (gitignore, khuyến nghị): thêm ASVP-ESD bản Kaggle `dejolilandry/asvpesdspeech-nonspeech-emotional-utterances` (13.964 WAV Audio + Bonus, giữ 12.227 file thuộc 9/13 nhãn) để bù surprise (5) từ 192 lên 1.778 và thêm 3 cảm xúc mới: contempt (7), boredom (8), excited (9). Tải bằng `kagglehub.dataset_download(...)` rồi chép vào `data/raw/asvp/` (xem `src/data/download_asvp.py`).
- `data/raw/affectnet_face/contempt/` (gitignore): ảnh khuôn mặt lớp contempt (7) cho nhánh mặt, crop sẵn 96x96.
- `data/raw/face_extra/boredom/` (gitignore): ảnh lớp boredom (8) từ Student Engagement, ảnh rộng → cần center-crop vuông.
- `data/raw/face_extra/excited/` (gitignore): ảnh lớp excited (9) — **hiện là proxy từ AffectNet `happy`**, chưa phải ground truth excited.
- `data/processed/face/` + `data/processed/audio/` (gitignore): output đã xử lý 48x48 và mel 128x128.
- `data/splits/` (gitignore): file `*_train/val/test.csv`, seed 42, audio split theo speaker. Cột audio: `filepath,speaker,label,source` (`source` là `ravdess`, `crema` hoặc `asvp`). Cột face: `label,pixels,Usage` (`Usage` là `Training`/`PublicTest`/`PrivateTest` cho FER2013 gốc, `extra` cho ảnh bổ sung).
- **Cả 2 nhánh dùng thang 10 lớp**: mặt 39.791 mẫu (train 31.441 / val 4.175 / test 4.175) gồm 35.887 FER2013 + 3.904 ảnh extra; tiếng 21.109 mẫu (train 14.457 / val 3.200 / test 3.452). Nhãn 0–9: `0 angry`, `1 disgust`, `2 fear`, `3 happy`, `4 sad`, `5 surprise`, `6 neutral`, `7 contempt`, `8 boredom`, `9 excited`.
- `data/processed/demo/` (được commit): 3–5 ảnh PNG + WAV mẫu nhẹ dưới 5 MB để demo 1 lệnh.
