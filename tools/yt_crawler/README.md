# 🇻🇳 Công cụ Cào & Gán Nhãn Dữ Liệu Cảm Xúc YouTube Tiếng Việt (`tools.yt_crawler`)

Module con hỗ trợ thu thập dữ liệu khuôn mặt + giọng nói **người Việt (Tiếng Việt)** từ YouTube để huấn luyện hoặc **Fine-tune** hệ thống `emotion-fusion`.

---

## 1. Luồng hoạt động (Pipeline)

```text
[Link YouTube / Từ khóa Tiếng Việt]
        │ (yt-dlp)
        ▼
[Video .mp4 + Audio 16kHz + Phụ đề Tiếng Việt .vtt]
        │ (extractor.py: Cắt cửa sổ 4.0s + Lọc khoảng lặng RMS + OpenCV Haar Face Crop)
        ▼
[Ảnh mặt 48x48 (.png) + Đoạn âm thanh 4s 16kHz (.wav) + Câu thoại Tiếng Việt]
        │ (pseudo_labeler.py: Gợi ý nhãn bằng Model CNN + Từ khóa cảm xúc Tiếng Việt + Âm học)
        ▼
[Giao diện duyệt nhãn Offline: data/processed/vn_youtube/review_labeler.html]
        │ (Người dùng nghe thử, bấm duyệt nhanh 0-6 và xuất metadata.csv)
        ▼
[export_splits.py: Chia tập 70/15/15 theo Speaker chống rò rỉ giọng]
        ▼
[data/splits/vn_face_{train,val,test}.csv & vn_audio_{train,val,test}.csv] ──► Fine-tune Model!
```

---

## 2. Hướng dẫn sử dụng từng bước

### Bước 1: Cào dữ liệu từ YouTube
Bạn có thể truyền danh sách link YouTube hoặc tìm kiếm trực tiếp bằng từ khóa tiếng Việt:

```bash
# Cách A: Truyền trực tiếp 1 hoặc nhiều link YouTube (phỏng vấn, vlog, phim ngắn Việt Nam)
python -m tools.yt_crawler crawl --urls "https://www.youtube.com/watch?v=VIDEO_ID_1" "https://www.youtube.com/watch?v=VIDEO_ID_2"

# Cách B: Tự động tìm và tải theo từ khóa tiếng Việt
python -m tools.yt_crawler crawl --query "phỏng vấn đường phố cảm xúc tiếng việt" --max-videos 3
```
*(Nếu bạn đã có model train sẵn từ FER2013/RAVDESS, thêm `--face-ckpt models/checkpoints/face/best.pt --audio-ckpt models/checkpoints/audio/best.pt` để dùng chính model đó gợi ý nhãn tự động!).*

### Bước 2: Kiểm tra & Chỉnh sửa nhãn trên giao diện HTML Offline
1. Sau khi chạy Bước 1, mở file sau bằng trình duyệt (chỉ cần double-click):
   👉 `data/processed/vn_youtube/review_labeler.html`
2. Trên giao diện:
   - Xem ảnh khuôn mặt `48x48`, đọc phụ đề tiếng Việt, bấm **Play** nghe đoạn giọng nói 4 giây.
   - Kiểm tra nhãn gợi ý tự động (`0: Tức giận` đến `6: Bình thường`). Nếu sai, click vào nút cảm xúc đúng hoặc bấm **🗑️ Loại bỏ** nếu đoạn cắt bị nhiễu.
3. Bấm nút **💾 Xuất file CSV đã duyệt (metadata.csv)** ở góc phải trên cùng và lưu đè vào `data/processed/vn_youtube/metadata.csv`.

### Bước 3: Xuất file Splits (Train / Val / Test)
Cập nhật lại tập chia sau khi đã duyệt nhãn:
```bash
python -m tools.yt_crawler split --csv data/processed/vn_youtube/metadata.csv
```
Lệnh này sẽ sinh ra:
- `data/splits/vn_face_train.csv`, `vn_face_val.csv`, `vn_face_test.csv`
- `data/splits/vn_audio_train.csv`, `vn_audio_val.csv`, `vn_audio_test.csv`

### Bước 4: Chạy Fine-tune trên dữ liệu Tiếng Việt
Sử dụng cờ `--pretrained-ckpt` và learning rate nhỏ (`--lr 0.0001`) để fine-tune tiếp từ model gốc:

```bash
# Fine-tune nhánh khuôn mặt
python -m src.training --branch face \
  --train-csv data/splits/vn_face_train.csv \
  --val-csv data/splits/vn_face_val.csv \
  --pretrained-ckpt models/checkpoints/face/best.pt \
  --lr 0.0001 --epochs 15

# Fine-tune nhánh giọng nói Tiếng Việt
python -m src.training --branch audio \
  --train-csv data/splits/vn_audio_train.csv \
  --val-csv data/splits/vn_audio_val.csv \
  --pretrained-ckpt models/checkpoints/audio/best.pt \
  --lr 0.0001 --epochs 15
```

---

## 3. Chạy thử ngay không cần mạng (Chế độ Mock)
Nếu muốn kiểm tra ngay giao diện `review_labeler.html` và luồng chia tập dữ liệu mà chưa cần tải video YouTube thật:
```bash
python -m tools.yt_crawler mock
```

