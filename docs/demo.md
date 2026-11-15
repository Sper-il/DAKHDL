# Demo — Chạy 1 lệnh Emotion Fusion

> Biến checkpoint tốt nhất thành demo chạy được ngay, sinh JSON + `index.html` rút gọn.

## Trạng thái: CHƯA LÀM

> Module này chưa được triển khai. Tài liệu giữ lại làm **đặc tả thiết kế** cho phần sắp xếp.
> File dưới đây đã tồn tại **rỗng (0 byte)** để thành viên mở ra viết.

File cần viết: `src/demo/__main__.py` — CLI 1 lệnh, nạp checkpoint, suy `num_classes` từ shape, dự đoán, ghi JSON + `index.html`.

## 1. Mục tiêu
- Fresh clone + 1 lệnh là chạy được, không cần mở notebook.
- Input mẫu nhẹ dưới 5 MB trong `data/processed/demo/`.
- Đo latency live, output đúng schema API đã chốt.

## 2. Input mẫu
```
data/processed/demo/
├── face_01.png   # ảnh xám 48x48 mẫu
└── voice_01.wav  # WAV 16 kHz mono mẫu 3–5 giây
```

Tiền xử lý demo: ảnh resize 48x48 grayscale, normalize mean 0,5 std 0,5; audio resample 16 kHz mono, pad/cắt 4 giây, mel 128x128, chuẩn hóa mean/std theo mẫu.

## 3. Lệnh chạy (sẽ khả dụng lại khi viết lại code)
```bash
# Chỉ nhánh mặt
python -m src.demo \
  --image data/processed/demo/face_01.png \
  --audio data/processed/demo/voice_01.wav \
  --face-ckpt models/checkpoints/face/best.pt

# Cả 2 nhánh + fusion
python -m src.demo \
  --image data/processed/demo/face_01.png \
  --audio data/processed/demo/voice_01.wav \
  --face-ckpt models/checkpoints/face/best.pt \
  --audio-ckpt models/checkpoints/audio/best.pt \
  --w-face 0.5 --w-audio 0.5
```

## 4. Output
```
results/demo/<ts>/
├── result.json  # {emotion, confidence, probs_fused[10], inference_time_ms}
└── index.html   # bản rút gọn của report.html, mở bằng double-click
```

Ví dụ `result.json`:
```json
{
  "emotion": "happy",
  "confidence": 0.87,
  "probs_fused": [0.02, 0.01, 0.01, 0.87, 0.03, 0.05, 0.01, 0.00, 0.00, 0.00],
  "inference_time_ms": 42.0
}
```

Đồng bộ khối Example Input → Output trong `README.md` với schema này.

## 5. Yêu cầu kỹ thuật khi viết lại
- **Suy `num_classes` từ shape checkpoint** thay vì hardcode 10 — đọc `backbone.fc.weight` (mặt) và `head.5.weight` (tiếng). Nhờ vậy tương thích cả checkpoint 7 lớp cũ lẫn 10 lớp mới, không crash `load_state_dict`.
- Label lấy từ `EMOTIONS` trong `src/models/fusion.py` (10 phần tử, cùng thứ tự với cả 2 nhánh).
- Có `--w-face` / `--w-audio` để chỉnh trọng số fusion khi demo.

## 6. Checklist nghiệm thu
- [ ] Chạy được trên máy sạch sau khi `pip install -r requirements.txt`.
- [ ] Output JSON đủ 4 trường, `probs_fused` tổng bằng 1 và **đủ 10 phần tử**.
- [ ] `index.html` mở offline được, hiện đúng emotion và latency.
- [ ] Latency end-to-end ≤ 200 ms trên GPU lab.

## 7. Output cũ còn trong repo (tham chiếu, KHÔNG phải kết quả hợp lệ)

`results/demo/20261109-1706/` là output của lần demo cũ, chạy trên model **7 lớp**:
- `probs_fused` chỉ có **7 phần tử** → sai schema 10 lớp hiện tại.
- `inference_time_ms: 11759.4` (11,7 giây) — vượt ngưỡng 200 ms rất xa, do chạy CPU lạnh chưa warm-up.

**Không được trích dẫn số này trong báo cáo.** Cần demo lại sau khi viết lại code và train full.
