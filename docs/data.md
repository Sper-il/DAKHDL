# Dữ liệu Emotion Fusion

- `data/raw/` (gitignore): đặt `fer2013.csv` và thư mục `audio/*.wav` tại đây, không sửa trực tiếp.
- `data/processed/face/` + `data/processed/audio/` (gitignore): output đã xử lý 48x48 và mel 128x128.
- `data/splits/` (gitignore): file `*_train/val/test.csv`, seed 42, audio split theo speaker.
- `data/processed/demo/` (được commit): 3–5 ảnh PNG + WAV mẫu nhẹ dưới 5 MB để demo 1 lệnh.
