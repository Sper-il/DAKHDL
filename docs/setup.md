# Setup — Cài đặt môi trường Emotion Fusion

> Hướng dẫn cài đặt duy nhất để chạy được project từ máy sạch. Đồng bộ với `requirements.txt`.

## 1. Yêu cầu hệ thống
- Python 3.10 trở lên (khuyến nghị 3.10 hoặc 3.11).
- Windows 10/11, Ubuntu 20.04+ hoặc macOS 13+.
- GPU NVIDIA có CUDA 11.8+ để train nhanh (fallback CPU-only vẫn chạy được data, eval và demo).
- Dung lượng trống tối thiểu 5 GB cho dữ liệu FER2013 + RAVDESS + checkpoint.

## 2. Cài đặt từ máy sạch
```bash
# 1. Clone + vào thư mục
cd emotion-fusion

# 2. Tạo môi trường ảo
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux/macOS

# 3. Nâng pip rồi cài dependency
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Nội dung `requirements.txt`: `torch`, `torchvision`, `albumentations`, `opencv-python`, `librosa`, `soundfile`, `scikit-learn`, `matplotlib`, `numpy`, `pandas`, `pyyaml`, `pytest`, `tensorboard`.

## 3. Kiểm tra cài đặt
```bash
python -c "import torch, librosa, albumentations; print(torch.__version__)"
pytest tests/unit -q
```

Kết quả mong đợi: `pytest` pass 3/3, `torch` in ra version 2.x, không lỗi import `librosa` hay `albumentations`.

## 4. Chuẩn bị dữ liệu
- Đặt `fer2013.csv` (hoặc `icml_face_data.csv`) vào `data/raw/`.
- Giải nén RAVDESS vào `data/raw/audio/` sao cho có 1.440 file WAV của 24 speaker.
- Chạy split:
```bash
python -m src.data.make_splits
```

Kiểm tra output trong `data/splits/`: `face_train.csv`, `face_val.csv`, `face_test.csv`, `audio_train.csv`, `audio_val.csv`, `audio_test.csv`.

## 5. Sự cố thường gặp
| Hiện tượng | Cách xử lý |
|------------|------------|
| `pip install` lỗi `soundfile` trên Windows | Cài `pip install pysoundfile`, hoặc dùng conda `conda install -c conda-forge libsndfile` |
| `torch` không nhận GPU | Cài lại bản CUDA đúng driver từ pytorch.org, hoặc chạy CPU-only để verify pipeline |
| `librosa` thiếu `numba` | `pip install numba==0.59.*` theo đúng Python version |
| Lỗi console mã tiếng Việt trên Windows | Chạy `chcp 65001` trước, code CLI đã dùng ASCII không dấu để tránh lỗi `charmap` |
