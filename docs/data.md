# Dữ liệu Emotion Fusion

- `data/raw/` (gitignore): đặt `fer2013.csv` và thư mục `audio/*.wav` tại đây, không sửa trực tiếp.
- `data/raw/crema/AudioWAV/` (gitignore, khuyến nghị): thêm 7.442 file WAV CREMA-D (91 speaker) để tăng đa dạng giọng nói. Tải bằng `python -m src.data.download_cremad` hoặc thủ công từ [Kaggle ejlok1/cremad](https://www.kaggle.com/datasets/ejlok1/cremad).
- `data/raw/asvp/{Audio,Bonus}/actor_X/*.wav` (gitignore, khuyến nghị): thêm ASVP-ESD bản Kaggle `dejolilandry/asvpesdspeech-nonspeech-emotional-utterances` (13.964 WAV Audio + Bonus, giữ 12.227 file thuộc 9/13 nhãn) để bù surprise (5) từ 192 lên 1.778 và thêm 3 cảm xúc mới: contempt (7), boredom (8), excited (9). Tải bằng `kagglehub.dataset_download(...)` rồi chép vào `data/raw/asvp/` (xem `src/data/download_asvp.py`).
- `data/processed/face/` + `data/processed/audio/` (gitignore): output đã xử lý 48x48 và mel 128x128.
- `data/splits/` (gitignore): file `*_train/val/test.csv`, seed 42, audio split theo speaker. Cột audio: `filepath,speaker,label,source` (`source` là `ravdess`, `crema` hoặc `asvp`). Nhánh mặt giữ 7 lớp, nhánh tiếng mở rộng 10 lớp (thêm `7 contempt — Khinh bỉ`, `8 boredom — Chán nản`, `9 excited — Hào hứng`).
- `data/processed/demo/` (được commit): 3–5 ảnh PNG + WAV mẫu nhẹ dưới 5 MB để demo 1 lệnh.
