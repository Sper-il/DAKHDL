# Demo — Chạy 1 lệnh Emotion Fusion

> Biến checkpoint tốt nhất thành demo chạy được ngay, sinh JSON + `index.html` rút gọn.

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

Tiền xử lý demo: ảnh resize 48x48 grayscale, normalize mean 0,5 std 0,5; audio resample 16 kHz mono, pad/cắt 4 giây, mel 128x128, chuẩn hóa mean/std theo mẫu. Xem `load_face_tensor` và `load_mel_tensor` trong `src/demo/__main__.py`.

## 3. Lệnh chạy
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
├── result.json  # {emotion, confidence, probs_fused[7], inference_time_ms}
└── index.html   # bản rút gọn của report.html, mở bằng double-click
```

Ví dụ `result.json`:
```json
{
  "emotion": "happy",
  "confidence": 0.87,
  "probs_fused": [0.02, 0.01, 0.01, 0.87, 0.03, 0.05, 0.01],
  "inference_time_ms": 42.0
}
```

Đồng bộ khối Example Input → Output trong `README.md` với schema này.

## 5. Checklist nghiệm thu
- [ ] Chạy được trên máy sạch sau khi `pip install -r requirements.txt`.
- [ ] Output JSON đủ 4 trường, `probs_fused` tổng bằng 1.
- [ ] `index.html` mở offline được, hiện đúng emotion và latency.
- [ ] Latency end-to-end ≤ 200 ms trên GPU lab.
