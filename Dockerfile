# Sử dụng base image Python 3.10 slim (ổn định, nhẹ và tương thích tốt với PyTorch, Librosa)
FROM python:3.10-slim

# Thiết lập biến môi trường
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

# Thiết lập thư mục làm việc trong container
WORKDIR /app

# Cài đặt các gói hệ thống cần thiết cho Librosa, SoundFile và OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libsndfile1 \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements trước để tận dụng Docker layer cache
COPY requirements.txt .

# Cài đặt các thư viện Python (mặc định PyTorch CPU, nếu chạy GPU xem ghi chú trong tài liệu)
RUN pip install --no-cache-dir -r requirements.txt

# Copy toàn bộ mã nguồn vào container
COPY . .

# Tạo sẵn các thư mục cần thiết nếu chưa có
RUN mkdir -p data/raw data/processed data/splits models/checkpoints models/trained results/runs results/demo

# Khởi chạy mặc định (hiển thị hướng dẫn hoặc chạy lệnh kiểm tra)
CMD ["python", "-m", "pytest", "tests/unit"]

