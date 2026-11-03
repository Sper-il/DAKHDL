# Decisions — Log quyết định kỹ thuật Emotion Fusion

> Mọi quyết định quan trọng đều ghi ở đây kèm lý do và phương án loại bỏ. Đồng bộ với mục Decisions Log trong `docs/progress_status.md`.

## 1. Khung ghi quyết định
```
[YYYY-MM-DD] Quyết định: <làm gì> | Lý do: <vì sao> | Loại bỏ: <phương án khác>
```

## 2. Quyết định đã chốt

### D-01: 7 cảm xúc cơ bản theo FER2013
- **Ngày**: 2026-11-01
- **Quyết định**: Giữ 7 lớp angry, disgust, fear, happy, sad, surprise, neutral.
- **Lý do**: Khớp FER2013 và RAVDESS, đủ khó để làm báo cáo nhưng vẫn đạt accuracy khả thi cho đồ án.
- **Loại bỏ**: 4 cảm xúc gọn (mất điểm đa dạng), 8 cảm xúc (thiếu dữ liệu contempt).

### D-02: FER2013 (mặt) + RAVDESS (tiếng)
- **Ngày**: 2026-11-01
- **Quyết định**: Mặt dùng FER2013 48x48 xám, tiếng dùng RAVDESS WAV 16 kHz.
- **Lý do**: Nhẹ, dễ tải, license nghiên cứu, đủ làm MVP trên GPU lab.
- **Loại bỏ**: AffectNet + IEMOCAP (nặng, license phức tạp, vượt scope MVP).

### D-03: Late fusion gộp xác suất
- **Ngày**: 2026-11-01
- **Quyết định**: `p_fused = w_face * p_face + w_audio * p_audio`, chuẩn hóa tổng bằng 1.
- **Lý do**: Hai nhánh độc lập dễ train, dễ ablation, đúng yêu cầu đa mô hình của đề tài.
- **Loại bỏ**: Early fusion (khó căn tốc độ học 2 modalities), chỉ làm mặt (không đạt yêu cầu đa mô hình).

### D-04: Local GPU + report.html offline
- **Ngày**: 2026-11-01
- **Quyết định**: Chạy batch offline, báo cáo một file HTML duy nhất không CDN.
- **Lý do**: Chấm bài không cần mạng, ít rủi ro latency, đúng chuẩn skill.
- **Loại bỏ**: Realtime webcam/micro (để Should/Could), Gradio Cloud (phụ thuộc mạng).

### D-05: Train riêng từng nhánh, best theo F1 val
- **Ngày**: 2026-11-01
- **Quyết định**: Trainer riêng face/audio, early stopping `patience=7` theo `val_f1_macro`.
- **Lý do**: Ổn định hơn train chung, chống mất cân bằng lớp disgust, reproduce từ config + seed 42.
- **Loại bỏ**: Train chung 2 nhánh (lệch gradient), chọn best theo loss train (dễ overfit).

### D-06: Split tiếng theo speaker, seed 42
- **Ngày**: 2026-11-01
- **Quyết định**: Audio split theo speaker (train 960 / val 180 / test 300), mặt giữ cột Usage gốc.
- **Lý do**: Chống leakage speaker, split cố định bằng file CSV trong `data/splits/`.
- **Loại bỏ**: Split random thuần (leakage speaker làm metric ảo).

### D-07: Mọi tài liệu nằm trong docs/, root chỉ giữ README.md
- **Ngày**: 2026-11-01
- **Quyết định**: `description.md`, `data.md`, `models.md`, `training.md`, `evaluation.md`, `visualization.md`, `demo.md`, `setup.md`, `runbook.md`, `decisions.md`, `progress_status.md`, `AI_Usage.md`, `CHANGELOG.md` đều nằm trong `docs/`.
- **Lý do**: Gọn root, dễ nộp bài, khớp quy ước người dùng chốt.
- **Loại bỏ**: Rải file `.md` ở root và `data/README.md` (khó tìm, dễ sót link).

## 3. Quyết định đang mở
- Trọng số fusion để 0,5/0,5 hay học trên val (`tune_on_val: true`)?
- Ngưỡng pass fusion ≥ 0,70 có cần hạ sau baseline không?
- Có làm nhánh realtime webcam/micro ở W5 hay giữ ở Could?
