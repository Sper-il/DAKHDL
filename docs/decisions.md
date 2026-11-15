# Decisions — Log quyết định kỹ thuật Emotion Fusion

> Mọi quyết định quan trọng đều ghi ở đây kèm lý do và phương án loại bỏ. Đồng bộ với mục Decisions Log trong `docs/progress_status.md`.

## 1. Khung ghi quyết định
```
[YYYY-MM-DD] Quyết định: <làm gì> | Lý do: <vì sao> | Loại bỏ: <phương án khác>
```

## 2. Quyết định đã chốt

### D-01: Cả 2 nhánh dùng thang 10 lớp (mở rộng từ 2026-11-15)
- **Ngày**: 2026-11-01 (rà soát 2026-11-09; **mở rộng 2026-11-15**)
- **Quyết định**: Thang 10 lớp chung: angry (0), disgust (1), fear (2), happy (3), sad (4), surprise (5), neutral (6), contempt (7), boredom (8), excited (9). Nhánh tiếng từ đầu đã có 10 lớp (RAVDESS + CREMA-D + ASVP-ESD). Từ 2026-11-15 **nhánh mặt cũng mở rộng lên 10 lớp** bằng 3.904 ảnh bổ sung; fusion gộp trực tiếp 10 lớp.
- **Lý do**: Đồ án đa mô hình cần cả 2 nhánh cùng thang để fusion không phải cắt bỏ 3 lớp mới. ASVP đã bù surprise từ 192 lên 1.778 mẫu; ảnh extra bù tiếp cho nhánh mặt (boredom chỉ 358, excited 600 mẫu).
- **Loại bỏ**: Giữ mặt 7 lớp / fusion 7 lớp (mất thông tin 3 lớp mới ở fusion); 4 cảm xúc gọn (mất điểm đa dạng); giữ đủ 13 nhãn ASVP (4 nhãn pleasure/pain/disappointment/breath không ánh xạ được).

### D-10: Nguồn ảnh cho 3 lớp mới của nhánh mặt
- **Ngày**: 2026-11-15
- **Quyết định**: contempt (7) từ AffectNet (crop sẵn 96x96, 2.946 ảnh); boredom (8) từ Student Engagement (358 ảnh, ảnh rộng → center-crop vuông); excited (9) **dùng ảnh `happy` từ AffectNet làm proxy** (600 ảnh).
- **Lý do**: Bổ sung đủ số lượng mẫu cho cả 3 lớp mới mà không cần thu thập thủ công.
- **Loại bỏ**: Chỉ dùng FER2013 (không có 3 lớp này); thu thập ảnh thật cho excited (tốn thời gian, ngoài scope) — đã ghi vào Open Questions làm giới hạn cần nói rõ trong báo cáo.
- **Rủi ro đã biết**: Ảnh khác domain FER2013, có thể ảnh hưởng accuracy 7 lớp gốc → cần ablation so sánh.

### D-11: Fusion gộp trực tiếp 10 lớp, `_align_probs` căn theo `min`
- **Ngày**: 2026-11-15
- **Quyết định**: Không cắt cứng về 7 lớp nữa. `_align_probs` dùng `min(len_face, len_audio)` → cả 2 nhánh 10 lớp thì giữ nguyên 10, chỉ fallback về 7 khi gặp checkpoint cũ.
- **Lý do**: 2 phân bố đã cùng thang và cùng thứ tự label nên không cần cắt; giữ fallback để không vỡ checkpoint cũ.
- **Loại bỏ**: Hardcode cắt 7 lớp đầu (mất 3 lớp mới), bỏ hẳn `_align_probs` (mất khả năng tương thích ckpt cũ).

### D-12: Suy `num_classes` từ shape checkpoint thay vì hardcode
- **Ngày**: 2026-11-15
- **Quyết định**: `demo/__main__.py` và `evaluation/__main__.py` đọc shape từ `backbone.fc.weight` (mặt) / `head.5.weight` (tiếng) để suy số lớp trước khi `load_state_dict`.
- **Lý do**: Không crash `load_state_dict` khi đổi số lớp; tương thích cả checkpoint 7 lớp cũ lẫn 10 lớp mới.
- **Loại bỏ**: Hardcode `num_classes=10` ở mọi nơi (vỡ khi đổi số lớp).

### D-02: FER2013 (mặt) + RAVDESS/CREMA-D/ASVP-ESD (tiếng)
- **Ngày**: 2026-11-01 (rà soát 2026-11-09)
- **Quyết định**: Mặt dùng FER2013 48x48 xám (35.887 mẫu); tiếng gộp 3 nguồn 21.109 WAV, 246 speaker (RAVDESS 1.440 + CREMA-D 7.442 + ASVP-ESD 12.227 file dùng được).
- **Lý do**: Nhẹ, dễ tải, license nghiên cứu, đủ làm MVP trên GPU lab.
- **Loại bỏ**: AffectNet + IEMOCAP (nặng, license phức tạp, vượt scope MVP).

### D-03: Late fusion gộp xác suất, căn lớp theo `min`
- **Ngày**: 2026-11-01 (rà soát 2026-11-09; **cập nhật 2026-11-15**)
- **Quyết định**: `p_fused = w_face * p_face_norm + w_audio * p_audio_norm`, căn lớp qua `_align_probs` rồi chuẩn hóa tổng bằng 1. Hiện cả 2 nhánh đều `[10]` → fusion `[10]` (xem D-11).
- **Lý do**: Hai nhánh độc lập dễ train, dễ ablation, đúng yêu cầu đa mô hình; căn lớp giữ shape không vỡ khi 2 nhánh lệch số lớp.
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
- **Quyết định**: Audio split theo speaker gộp 3 nguồn (train 14.457 / 172 speaker, val 3.200 / 36 speaker, test 3.452 / 38 speaker). Mặt giữ cột Usage gốc (28.709 / 3.589 / 3.589) **+ 3.904 ảnh extra chia 70/15/15** (train 31.441 / val 4.175 / test 4.175). Giao speaker train/val/test rỗng.
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
- Cần thu thập ảnh `excited` thật thay cho proxy từ `happy` không?
- Ảnh extra có làm giảm accuracy 7 lớp FER2013 gốc không — cần ablation so sánh.
