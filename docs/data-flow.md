# Luồng dữ liệu Emotion Fusion

## Mặt (FER2013)
`data/raw/fer2013.csv` → parse pixel 2304 số → ảnh 48x48 → `data/processed/face/` (PNG) → `data/splits/face_{train,val,test}.csv` (70/15/15, seed 42, stratify theo label) → `FaceDataset → {image [1,48,48], label}`.

Tiền xử lý: resize 48x48 (giữ nguyên), normalize mean 0,5 std 0,5. Augment chỉ train: flip ngang, xoay ±10 độ, brightness/contrast nhẹ (Albumentations).

## Tiếng (RAVDESS/CREMA-D)
`data/raw/audio/*.wav` → resample 16 kHz mono → cắt/pad 4 giây → Mel-spectrogram 128 mel x 128 khung → log-mel normalize → `data/processed/audio/` (npy) → `data/splits/audio_{train,val,test}.csv` (split theo speaker, seed 42) → `AudioDataset → {mel [1,128,128], label}`.

Augment chỉ train: time mask + freq mask (SpecAugment), thêm nhiễu trắng nhẹ.

## Train → Eval → Report
`processed + splits` → train riêng từng nhánh → `models/checkpoints/` → best theo F1 val → eval test → `results/runs/<ts>/{metrics.json, plots/, report.html}`. Demo đọc checkpoint tốt nhất, chạy trên `data/processed/demo/` → `results/demo/<ts>/{index.html, metrics.json}`.
