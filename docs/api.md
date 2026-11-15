# API nội bộ giữa các module

## FaceDataset
- Input: `face_01.png` (xám 48x48) + label 0–9 (7 lớp FER2013 + contempt/boredom/excited).
- Output: `{'image': tensor [1,48,48], 'label': int}`.

## AudioDataset
- Input: `voice_01.wav` (16 kHz mono ~4 giây) + label 0–9 (7 lớp chung + contempt/boredom/excited).
- Output: `{'mel': tensor [1,128,128], 'label': int}`.
- Nguồn nhanh: `data/splits/audio_*_proc.csv` (cột `mel_path`) để đọc mel `.npy` có sẵn, không cần librosa lúc train.

## FaceCNN / AudioCNN
- Input: batch `[B,1,48,48]` (mặt) hoặc `[B,1,128,128]` (tiếng).
- Output: `probs_face [B,10]` sau softmax; `probs_audio [B,10]` sau softmax.

## fuse_probs
- Input: `p_face [10] hoặc [B,10]`, `p_audio [10] hoặc [B,10]`, `w_face=0,5, w_audio=0,5`.
- Xử lý: `_align_probs` cắt về `min` số lớp của 2 phân bố (cả 2 đều 10 nên giữ nguyên 10; fallback 7 cho checkpoint cũ) rồi chuẩn hoá lại tổng bằng 1.
- Output: `p_fused [10]`, `emotion: str`, `confidence: float`.
- Công thức: `p_fused = w_face * p_face_norm + w_audio * p_audio_norm`, chuẩn hóa tổng bằng 1.

## visualize_html.build_report
> Trạng thái: chưa triển khai. Xem `docs/visualization.md`.
- Input (thiết kế): `run_dir` chứa `metrics.json`, `config.yaml`, `plots/*.png`.
- Output (thiết kế): `report.html` offline duy nhất, số khớp `metrics.json`.

## demo CLI
> Trạng thái: chưa triển khai. Xem `docs/demo.md`.
- Lệnh (thiết kế): `python -m src.demo --image <png> --audio <wav> --face-ckpt <pt> --audio-ckpt <pt>`.
- Output (thiết kế): JSON emotion + `results/demo/<ts>/index.html`.
- Yêu cầu kỹ thuật: suy `num_classes` tự động từ shape checkpoint (`backbone.fc.weight` cho mặt, `head.5.weight` cho tiếng) để tương thích cả checkpoint 7 lớp cũ lẫn 10 lớp mới.
