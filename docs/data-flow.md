# Luồng dữ liệu Emotion Fusion

## Mặt (FER2013 + ảnh extra cho lớp mới)
`data/raw/fer2013.csv` → parse pixel 2304 số → ảnh 48x48 → `data/processed/face/` (PNG) → `data/splits/face_{train,val,test}.csv` (70/15/15, seed 42, stratify theo label) → `FaceDataset → {image [1,48,48], label 0-9}`.

Tiền xử lý: resize 48x48 (giữ nguyên), normalize mean 0,5 std 0,5. Augment chỉ train: flip ngang, xoay ±10 độ, brightness/contrast nhẹ (Albumentations).

**Bổ sung ảnh cho 3 lớp mới** (`python -m src.data.add_face_extra`): `data/raw/affectnet_face/contempt/` → label 7, `data/raw/face_extra/boredom/` → label 8 (ảnh rộng, center-crop vuông), `data/raw/face_extra/excited/` → label 9 (proxy từ ảnh `happy`). Ảnh được resize về 48x48, chuyển thành chuỗi pixel để **nối thẳng vào file split hiện có** (cùng định dạng `label,pixels,Usage`, `Usage=extra`), giữ nguyên 35.887 mẫu FER2013. Chia extra 70/15/15 riêng (stratify theo label, seed 42) rồi gộp vào split tương ứng. Kết quả: **39.791 mẫu 10 lớp** — train 31.441 / val 4.175 / test 4.175.

## Tiếng (RAVDESS + CREMA-D + ASVP-ESD bổ sung)
`data/raw/audio/*.wav` (RAVDESS, 1.440 file, 24 speaker) + `data/raw/crema/AudioWAV/*.wav` (CREMA-D, 7.442 file, 91 speaker, nếu có) + `data/raw/asvp/{Audio,Bonus}/actor_X/*.wav` (ASVP-ESD bản Kaggle, tới 13.964 WAV, giữ 12.227 thuộc 9/13 nhãn) → resample 16 kHz mono → cắt/pad 4 giây → Mel-spectrogram 128 mel x 128 khung → log-mel normalize → `data/processed/audio/` (npy) → `data/splits/audio_{train,val,test}.csv` (gộp 3 nguồn, split theo speaker, seed 42) → `data/splits/audio_*_proc.csv` → `AudioDataset → {mel [1,128,128], label 0-9}`. Nhãn tiếng 0–9 giống hệt nhánh mặt: 7 gốc chung + `7 contempt`, `8 boredom`, `9 excited` (chỉ có ở ASVP).

Augment chỉ train: time mask + freq mask (SpecAugment), thêm nhiễu trắng nhẹ.

## Train → Eval → Report
`processed + splits` → train riêng từng nhánh → `models/checkpoints/` → best theo F1 val → eval test → `results/runs/<ts>/{metrics.json, plots/, report.html}`. Demo đọc checkpoint tốt nhất, chạy trên `data/processed/demo/` → `results/demo/<ts>/{index.html, metrics.json}`.
