# Decisions — Log quyết định kỹ thuật Emotion Fusion

> Mọi quyết định quan trọng đều ghi ở đây kèm lý do và phương án loại bỏ. Đồng bộ với mục Decisions Log trong `docs/progress_status.md`.

## 1. Khung ghi quyết định
```
[YYYY-MM-DD] Quyết định: <làm gì> | Lý do: <vì sao> | Loại bỏ: <phương án khác>
```

## 2. Quyết định đã chốt

### D-01: 7 lớp mặt theo FER2013, tiếng mở rộng 10 lớp
- **Ngày**: 2026-11-01 (rà soát 2026-11-09 theo dữ liệu thật 21.109 WAV)
- **Quyết định**: Nhánh mặt giữ 7 lớp angry, disgust, fear, happy, sad, surprise, neutral; nhánh tiếng mở rộng 10 lớp, thêm contempt (7), boredom (8), excited (9) chỉ có ở ASVP-ESD; fusion căn về 7 lớp chung.
- **Lý do**: Khớp FER2013 và RAVDESS/CREMA-D/ASVP, đủ khó để làm báo cáo nhưng vẫn đạt accuracy khả thi; ASVP đã bù surprise từ 192 lên 1.778 mẫu.
- **Loại bỏ**: 4 cảm xúc gọn (mất điểm đa dạng), giữ đủ 13 nhãn ASVP (4 nhãn pleasure/pain/disappointment/breath không ánh xạ được).

### D-02: FER2013 (mặt) + RAVDESS/CREMA-D/ASVP-ESD (tiếng)
- **Ngày**: 2026-11-01 (rà soát 2026-11-09)
- **Quyết định**: Mặt dùng FER2013 48x48 xám (35.887 mẫu); tiếng gộp 3 nguồn 21.109 WAV, 246 speaker (RAVDESS 1.440 + CREMA-D 7.442 + ASVP-ESD 12.227 file dùng được).
- **Lý do**: Nhẹ, dễ tải, license nghiên cứu, đủ làm MVP trên GPU lab.
- **Loại bỏ**: AffectNet + IEMOCAP (nặng, license phức tạp, vượt scope MVP).

### D-03: Late fusion gộp xác suất, căn về 7 lớp chung
- **Ngày**: 2026-11-01 (rà soát 2026-11-09)
- **Quyết định**: `p_fused = w_face * p_face_norm + w_audio * p_audio_norm`, cắt 7 lớp đầu qua `_align_probs` rồi chuẩn hóa tổng bằng 1 (mặt `[7]`, tiếng `[10]` → fusion `[7]`).
- **Lý do**: Hai nhánh độc lập dễ train, dễ ablation, đúng yêu cầu đa mô hình; tiếng bắt buộc 10 lớp để giữ contempt/boredom/excited nên phải căn lớp mới không vỡ shape.
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
- **Ngày**: 2026-11-01 (rà soát 2026-11-09)
- **Quyết định**: Audio split theo speaker gộp 3 nguồn (train 14.457 / 172 speaker, val 3.200 / 36 speaker, test 3.452 / 38 speaker), mặt giữ cột Usage gốc (28.709 / 3.589 / 3.589). Giao speaker train/val/test rỗng.
- **Lý do**: Chống leakage speaker, split cố định bằng file CSV trong `data/splits/`.
- **Loại bỏ**: Split random thuần (leakage speaker làm metric ảo).

### D-08: ASVP-ESD lấy bản Kaggle, mã cảm xúc ở token thứ 3
- **Ngày**: 2026-11-09
- **Quyết định**: Dùng bản Kaggle `dejolilandry/asvpesdspeech-nonspeech-emotional-utterances` (13.964 WAV `Audio/` + `Bonus/`), mã cảm xúc đọc ở token thứ 3 dạng `03-<01 Speech|02 NonSpeech>-<01..13>-...`, chỉ giữ 9/13 nhãn.
- **Lý do**: Script Hugging Face `EdwardLin2023/ASVP_ESD` đã deprecated (`load_dataset` báo lỗi), bản Kaggle đọc được offline và đủ 12.227 file dùng được.
- **Loại bỏ**: Tải qua `datasets.load_dataset` (lỗi script deprecated), đọc mã ở 2 ký tự đầu (sai với format Kaggle).

### D-09: Đọc mel nhanh qua `audio_*_proc.csv`
- **Ngày**: 2026-11-09
- **Quyết định**: Train đọc mel `.npy` có sẵn qua cột `mel_path`, không đọc WAV on-the-fly bằng librosa lúc train.
- **Lý do**: Nhanh hơn, không phụ thuộc librosa lúc train, đã kiểm chứng thiếu 0/21.109 file.

### D-07: Mọi tài liệu nằm trong docs/, root chỉ giữ README.md
- **Ngày**: 2026-11-01
- **Quyết định**: `description.md`, `data.md`, `models.md`, `training.md`, `evaluation.md`, `visualization.md`, `demo.md`, `setup.md`, `runbook.md`, `decisions.md`, `progress_status.md`, `AI_Usage.md`, `CHANGELOG.md` đều nằm trong `docs/`.
- **Lý do**: Gọn root, dễ nộp bài, khớp quy ước người dùng chốt.
- **Loại bỏ**: Rải file `.md` ở root và `data/README.md` (khó tìm, dễ sót link).

## 3. Quyết định đang mở
- Trọng số fusion để 0,5/0,5 hay học trên val (`tune_on_val: true`)?
- Ngưỡng pass fusion ≥ 0,70 có cần hạ sau baseline không?
- Có làm nhánh realtime webcam/micro ở W5 hay giữ ở Could?
