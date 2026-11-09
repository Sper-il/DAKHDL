# Dữ liệu Emotion Fusion

- `data/raw/` (gitignore): đặt `fer2013.csv` và thư mục `audio/*.wav` tại đây, không sửa trực tiếp.
- `data/raw/crema/AudioWAV/` (gitignore, khuyến nghị): thêm 7.442 file WAV CREMA-D (91 speaker) để tăng đa dạng giọng nói. Tải bằng `python -m src.data.download_cremad` hoặc thủ công từ [Kaggle ejlok1/cremad](https://www.kaggle.com/datasets/ejlok1/cremad).
- `data/raw/asvp/{Speech,NonSpeech}/actor_X/*.wav` (gitignore, khuyến nghị): thêm ASVP-ESD (2.150 clip Speech + 5.484 clip NonSpeech, 13 nhãn gốc) để bù 4 nhóm cảm xúc mới: surprise (5), contempt (7), boredom (8), excited (9). Tải bằng `python -m src.data.download_asvp` (nguồn Hugging Face `EdwardLin2023/ASVP_ESD`, không cần key).
- `data/processed/face/` + `data/processed/audio/` (gitignore): output đã xử lý 48x48 và mel 128x128.
- `data/splits/` (gitignore): file `*_train/val/test.csv`, seed 42, audio split theo speaker. Cột audio: `filepath,speaker,label,source` (`source` là `ravdess`, `crema` hoặc `asvp`). Nhánh mặt giữ 7 lớp, nhánh tiếng mở rộng 10 lớp (thêm `7 contempt — Khinh bỉ`, `8 boredom — Chán nản`, `9 excited — Hào hứng`).
- `data/processed/demo/` (được commit): 3–5 ảnh PNG + WAV mẫu nhẹ dưới 5 MB để demo 1 lệnh.
