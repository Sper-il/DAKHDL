# Dữ liệu Emotion Fusion

- `data/raw/` (gitignore): đặt `fer2013.csv` và thư mục `audio/*.wav` tại đây, không sửa trực tiếp.
- `data/raw/crema/AudioWAV/` (gitignore, khuyến nghị): thêm 7.442 file WAV CREMA-D (91 speaker) để tăng đa dạng giọng nói. Tải bằng `python -m src.data.download_cremad` hoặc thủ công từ [Kaggle ejlok1/cremad](https://www.kaggle.com/datasets/ejlok1/cremad).
- `data/processed/face/` + `data/processed/audio/` (gitignore): output đã xử lý 48x48 và mel 128x128.
- `data/splits/` (gitignore): file `*_train/val/test.csv`, seed 42, audio split theo speaker. Cột audio: `filepath,speaker,label,source` (`source` là `ravdess` hoặc `crema`).
- `data/processed/demo/` (được commit): 3–5 ảnh PNG + WAV mẫu nhẹ dưới 5 MB để demo 1 lệnh.
