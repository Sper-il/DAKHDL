# API nội bộ giữa các module

## FaceDataset
- Input: `face_01.png` (xám 48x48) + label 0–6.
- Output: `{'image': tensor [1,48,48], 'label': int}`.

## AudioDataset
- Input: `voice_01.wav` (16 kHz mono ~4 giây) + label 0–6.
- Output: `{'mel': tensor [1,128,128], 'label': int}`.

## FaceCNN / AudioCNN
- Input: batch `[B,1,48,48]` (mặt) hoặc `[B,1,128,128]` (tiếng).
- Output: `probs [B,7]` sau softmax.

## fuse_probs
- Input: `p_face [7], p_audio [7], w_face=0,5, w_audio=0,5`.
- Output: `p_fused [7]`, `emotion: str`, `confidence: float`.
- Công thức: `p_fused = w_face * p_face + w_audio * p_audio`, chuẩn hóa tổng bằng 1.

## visualize_html.build_report
- Input: `run_dir` chứa `metrics.json`, `config.yaml`, `plots/*.png`.
- Output: `report.html` offline duy nhất, số khớp `metrics.json`.

## demo CLI
- Lệnh: `python -m src.demo --image <png> --audio <wav> --face-ckpt <pt> --audio-ckpt <pt>`.
- Output: JSON emotion + `results/demo/<ts>/index.html`.
