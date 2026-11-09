# Hệ thống nhận diện cảm xúc con người dựa trên biểu cảm khuôn mặt và tín hiệu âm thanh sử dụng Deep Learning

> Single source of truth cho **đang làm gì, tại sao, và đã làm được gì**.
> Cập nhật lần cuối: 2026-11-01

## Project Context (Phase 1 — DEFINE)
| Trường | Giá trị |
|--------|---------|
| Tên đầy đủ | Hệ thống nhận diện cảm xúc con người dựa trên biểu cảm khuôn mặt và tín hiệu âm thanh sử dụng Deep Learning |
| Slug | `emotion-fusion` |
| Tagline | Nhận diện 7 cảm xúc cơ bản từ ảnh khuôn mặt và giọng nói bằng Deep Learning với fusion muộn, báo cáo bằng một file HTML offline. |
| Bài toán | Phân loại (classification) 7 cảm xúc cơ bản: angry, disgust, fear, happy, sad, surprise, neutral từ 2 modalities: ảnh khuôn mặt và đoạn âm thanh giọng nói. Quan trọng vì cảm xúc là tín hiệu cốt lõi trong tương tác người–máy, hỗ trợ giáo dục, tư vấn và chăm sóc khách hàng. |
| Người dùng mục tiêu | Sinh viên và giảng viên chấm đồ án; nhà nghiên cứu AI muốn baseline đa mô hình gọn nhẹ. Bối cảnh: chạy local GPU trong phòng lab, xem kết quả qua `report.html` và demo 1 lệnh. Kỳ vọng output: nhãn cảm xúc + phân bố xác suất + độ trễ suy luận. |
| Input | Nhánh mặt: ảnh xám 48x48 (`FER2013`, `data/raw/fer2013.csv`). Nhánh tiếng: file WAV 16 kHz mono 3–5 giây (`RAVDESS`/`CREMA-D`), chuyển thành Mel-spectrogram 128x128. Demo dùng ảnh PNG + WAV mẫu trong `data/processed/demo/`. |
| Output | JSON: `{ emotion, confidence, probs_face[7], probs_audio[7], probs_fused[7], inference_time_ms }` + `report.html` trong `results/runs/<ts>/` và `index.html` rút gọn trong `results/demo/<ts>/`. |
| Tính năng cốt lõi | Must: data pipeline mặt + tiếng, model CNN mặt, model CNN âm thanh, late fusion, eval 2 lớp, `report.html` offline, demo 1 lệnh. Should: webcam + micro realtime, confusion matrix per-class, `index.html` demo. Could: export ONNX, Gradio UI, API FastAPI. Out of scope MVP: nhận diện nhiều người cùng lúc, phân tích video dài, cảm xúc phức tạp (contempt, stress). |
| Tech construction | PyTorch 2.x, torchvision, Albumentations, librosa, scikit-learn, matplotlib (vẽ PNG rồi nhúng base64, không CDN), TensorBoard. Chạy local GPU (fallback CPU-only), batch offline. |
| Tiêu chí đánh giá | Model level: Accuracy ≥ 0,65 (mặt FER2013), ≥ 0,70 (tiếng RAVDESS), F1 macro ≥ 0,60, fusion tốt hơn từng nhánh ≥ 3 điểm. Pipeline: latency end-to-end ≤ 200 ms/mẫu trên GPU, `report.html` < 10 MB, mở offline được. |

## Architecture Snapshot (Phase 2 — DESIGN)
- System architecture: late fusion 2 nhánh độc lập (Face-CNN + Audio-CNN) rồi gộp xác suất có trọng số, chi tiết xem `docs/architecture.md`.
- Modules: `data` (load/augment/split mặt + tiếng) | `models` (face_cnn, audio_cnn, fusion) | `training` (trainer 2 nhánh + fusion) | `evaluation` (metric 2 lớp + `visualize_html.py`) | `demo` (script 1 lệnh + notebook). Trách nhiệm chi tiết xem `docs/api.md`.
- Folder structure: theo chuẩn skill (`templates/folder-structure.md`), đã scaffold đủ `data/`, `src/`, `models/`, `configs/`, `results/runs/`, `results/demo/`, `docs/`.
- Data flow: `raw (fer2013.csv + wav) → processed (face 48x48 + mel 128x128) → splits (70/15/15, seed 42) → train → predictions → results/runs/<ts>/{metrics.json, report.html}` (tần suất batch offline, chi tiết xem `docs/data-flow.md`).
- Interfaces / APIs: `FaceDataset → {image [1,48,48], label}`; `AudioDataset → {mel [1,128,128], label}`; `FaceCNN → probs[7]`; `AudioCNN → probs[7]`; `fuse(p_face, p_audio, w) → probs_fused[7]`; `visualize_html.build_report(run_dir) → report.html`. Chi tiết xem `docs/api.md`.
- Milestones: W1: scaffold + Define/Design (hiện tại) | W2: data pipeline mặt + tiếng | W3: model 2 nhánh + train baseline | W4: late fusion + eval + `report.html` | W5: demo 1 lệnh + README + nộp bài.
- Technical risks: overfit FER2013 do mất cân bằng (disgust ít) → weighted loss + augment; leakage speaker giữa train/val tiếng → split theo speaker; VRAM thiếu khi train 2 nhánh → train riêng từng nhánh, batch nhỏ; xem bảng `Risks & Mitigations` bên dưới.

## Active Module: training + evaluation + demo — sanity xong, chờ train full
- **Describe** — Train riêng từng nhánh (Trainer + early stopping F1 val), eval 2 lớp + render `report.html`, demo 1 lệnh sinh `index.html`.
- **Goal / purpose** — Có pipeline chạy được từ split tới báo cáo HTML ngay khi có FER2013 + RAVDESS.
- **Pipeline** — `splits → trainer.fit() → checkpoints → evaluation → metrics.json + plots + report.html → demo → results/demo/<ts>/`.
- **How / experiment plan (sub-tasks)**
  - [x] `trainer.py` (`train_one_epoch`, `validate`, `save_checkpoint`, early stopping `val_f1_macro`) + `training/__main__.py` CLI theo branch.
  - [x] `evaluation/metrics.py` (`evaluate_branch`, `evaluate_fusion`) + `visualize.py` + `visualize_html.py` + `evaluation/__main__.py`.
  - [x] `demo/__main__.py` 1 lệnh (ảnh + wav → JSON + `index.html`), đo latency live.
  - [x] Imports OK, `pytest tests/unit` pass 4/4, không lỗi lint.
  - [x] Data thật đã có: mặt train 28709 / val 3589 / test 3589 (theo cột Usage gốc), tiếng train 960 / val 180 / test 300 (split theo speaker, seed 42). Loader thật pass shape `[4,1,48,48]` và `[4,1,128,128]`.
  - [x] Dữ liệu demo nhẹ `data/processed/demo/` (3 PNG + 3 WAV, ~1,1 MB) qua `python -m src.data.make_demo`.
  - [x] Train sanity nhánh mặt + tiếng (`--sanity`, CPU): tiếng train_loss 1,87→1,52, mặt train_loss 2,36→1,41 — loss giảm, sinh `best.pt` 2 nhánh.
  - [x] Eval sanity `results/runs/20261109-1704/`: mặt acc 0,18 / F1 0,11, tiếng acc 0,39 / F1 0,28 (chưa đạt target, đúng kỳ vọng vì chỉ 2 epoch trên 32 mẫu).
  - [x] Demo 1 lệnh `results/demo/20261109-1706/`: ra `neutral` conf 0,25, JSON đủ `{emotion, confidence, probs_fused[7], inference_time_ms}`.
  - [ ] Train full 30 epoch trên GPU lab rồi eval lại để đạt target (mặt ≥ 0,65, tiếng ≥ 0,70).
- **Status** — Describe ✅ / Generate ✅ / Review ✅ / Run ✅ (sanity + eval + demo thật) / Test ✅ (4/4 + loader thật) / Refactor ✅ / Commit ⏳
- **Last run** — `python -m src.training --branch audio/face --sanity` xong; `python -m src.evaluation` ra `metrics.json` + `report.html`; `python -m src.demo` ra `result.json` + `index.html`.
- **DoD checklist** — xem `checklists/dod.md` (workspace skill)

## Module Log
| # | Module | Ngày | Status | Commit | Ghi chú |
|---|--------|------|--------|--------|---------|
| 0 | Scaffold + Define/Design | 2026-11-01 | ✅ | — | folders + docs/progress_status + README + docs/AI_Usage + docs + configs + docs/description.md |
| 1 | data (mặt + tiếng) | 2026-11-01 | ✅ | — | FER 35.887 mẫu + RAVDESS 1440 WAV 24 speaker, split seed 42, loader thật pass shape |
| 2 | models (face_cnn + audio_cnn + fusion) | 2026-11-01 | ✅ | — | forward đúng [B,7], fusion chuẩn hóa tổng 1, face ~11,17M + audio ~1,21M, pytest 3/3 |
| 3 | training + evaluation + visualization + demo | 2026-11-01 | ✅ code, ⏳ train thật | — | trainer + eval 2 lớp + report.html + demo 1 lệnh, imports OK, pytest 3/3 |
| 4 | docs bổ sung | 2026-11-01 | ✅ | — | thêm models, training, evaluation, visualization, demo, setup, runbook, decisions, README index trong docs/ |
| 5 | EDA dữ liệu | 2026-11-03 | ✅ | — | script notebooks/2026-11-03_eda.py, plots results/eda/ (6 biểu đồ phân bố + mẫu mặt/mel), docs/eda.md |

## Decisions Log
| Ngày | Quyết định | Lựa chọn khác | Lý do |
|------|------------|---------------|-------|
| 2026-11-01 | 7 cảm xúc cơ bản theo FER2013 | 4 cảm xúc gọn / 8 cảm xúc | Khớp FER2013 + RAVDESS, đủ khó để làm báo cáo nhưng vẫn đạt accuracy khả thi. |
| 2026-11-01 | FER2013 (mặt) + RAVDESS/CREMA-D (tiếng) | AffectNet + IEMOCAP | Nhẹ, dễ tải, có license nghiên cứu, đủ làm MVP trên GPU lab. |
| 2026-11-01 | Late fusion gộp xác suất | Early fusion / chỉ làm mặt | Hai nhánh độc lập dễ train, dễ so sánh, đúng yêu cầu đa mô hình của đề tài. |
| 2026-11-01 | Local GPU + `report.html` offline | Realtime webcam + micro | Đúng chuẩn skill, chấm bài không cần mạng, ít rủi ro latency. |

## Risks & Mitigations
| Rủi ro | Khả năng | Tác động | Giảm thiểu | Phát hiện |
|--------|----------|----------|------------|-----------|
| FER2013 mất cân bằng (disgust rất ít) | Cao | Trung bình | Weighted cross-entropy + augment mặt, báo F1 macro | Phân bố lớp trong report |
| Leakage speaker ở nhánh tiếng | Trung bình | Cao | Split theo speaker, seed 42, lưu file split | So sánh accuracy val/test |
| VRAM thiếu khi train 2 nhánh | Trung bình | Trung bình | Train riêng từng nhánh, batch 32/16, mixed precision | Log GPU trong train |
| Audio nhiễu nền làm metric thấp | Trung bình | Trung bình | Chuẩn hóa 16 kHz, cắt im lặng, augment SpecAugment | Nghe thử + xem mel mẫu |

## Open Questions
- [ ] Link tải FER2013 và RAVDESS chính thức dùng trong lab (Kaggle hay repo trường)?
- [ ] Trọng số late fusion để mặc định 0,5/0,5 hay học trên val?
- [ ] Ngưỡng pass cuối: Accuracy fusion ≥ 0,70 có cần điều chỉnh sau baseline không?
- [ ] Có cần nhánh realtime webcam/micro ở milestone W5 hay để Could?

## Definition of Done (áp dụng mỗi module)
- [ ] Đúng yêu cầu
- [ ] Code chạy end-to-end
- [ ] Tests pass
- [ ] Không có secrets / không có artifact lớn
- [ ] Code đã self-review
- [ ] Có thể giải thích được
- [ ] Docs đã cập nhật
- [ ] Đã committed (Conventional Commit)
