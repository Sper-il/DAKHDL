# Hệ thống nhận diện cảm xúc con người dựa trên biểu cảm khuôn mặt và tín hiệu âm thanh sử dụng Deep Learning

> Single source of truth cho **đang làm gì, tại sao, và đã làm được gì**.
> Cập nhật lần cuối: 2026-11-09 (rà soát lần 2 theo dữ liệu thực tế 21.109 WAV)

## Project Context (Phase 1 — DEFINE)
| Trường | Giá trị |
|--------|---------|
| Tên đầy đủ | Nghiên cứu và xây dựng hệ thống nhận diện cảm xúc con người dựa trên biểu cảm khuôn mặt và tín hiệu âm thanh sử dụng Deep Learning |
| Slug | `emotion-fusion` |
| Tagline | Nhận diện 7 cảm xúc chung từ ảnh khuôn mặt và giọng nói bằng Deep Learning với fusion muộn; nhánh tiếng mở rộng 10 lớp nhờ ASVP-ESD, báo cáo bằng một file HTML offline. |
| Bài toán | Phân loại (classification) cảm xúc từ 2 modalities. Nhánh mặt giữ 7 lớp theo FER2013: angry (0), disgust (1), fear (2), happy (3), sad (4), surprise (5), neutral (6). Nhánh tiếng mở rộng 10 lớp: 7 lớp chung cộng thêm contempt (7), boredom (8), excited (9) chỉ có ở ASVP-ESD. Khi fusion, 2 phân bố được căn về 7 lớp chung qua `_align_probs` rồi cộng có trọng số. Quan trọng vì cảm xúc là tín hiệu cốt lõi trong tương tác người–máy, hỗ trợ giáo dục, tư vấn và chăm sóc khách hàng. ASVP-ESD đã bù điểm nghẽn thực tế: lớp `surprise` từ 192 mẫu (2,2%) lên 1.778 mẫu (8,4%). |
| Người dùng mục tiêu | Sinh viên thực hiện đồ án và giảng viên chấm bài; nhà nghiên cứu AI cần baseline đa mô hình gọn nhẹ. Bối cảnh: chạy local GPU trong phòng lab (fallback CPU-only), xem kết quả qua `report.html` và `results/eda/report.html`, demo 1 lệnh. Kỳ vọng output: nhãn cảm xúc + phân bố xác suất (mặt 7, tiếng 10, fusion 7) + độ trễ suy luận. |
| Input | Nhánh mặt: ảnh xám 48x48, nguồn FER2013 (`data/raw/fer2013.csv`, 35.887 mẫu: train 28.709 / val 3.589 / test 3.589 theo cột `Usage` gốc). Nhánh tiếng: file WAV 16 kHz mono 3–5 giây, chuẩn hoá về 4 giây thành Mel 128x128; nguồn gộp 3 bộ tổng 21.109 WAV, 246 speaker (RAVDESS 1.440 + CREMA-D 7.442 + ASVP-ESD 12.227 file dùng được), split theo speaker seed 42: train 14.457 (172 speaker) / val 3.200 (36) / test 3.452 (38), giao nhau rỗng. Demo dùng ảnh PNG + WAV mẫu trong `data/processed/demo/`. |
| Output | JSON: `{ emotion, confidence, probs_face[7], probs_audio[10], probs_fused[7], inference_time_ms }` + `report.html` trong `results/runs/<ts>/` và `index.html` rút gọn trong `results/demo/<ts>/`. |
| Tính năng cốt lõi | Must: data pipeline mặt (48x48) + tiếng (mel 128x128, đủ 21.109 `.npy`), Face-CNN 7 lớp + Audio-CNN 10 lớp + late fusion căn 7 lớp chung, eval 2 lớp, `report.html` offline, demo 1 lệnh. Should: confusion matrix per-class (mặt 7x7, tiếng 10x10, fusion 7x7) + 5 case tệ nhất, webcam/micro realtime, `index.html` demo. Could: export ONNX, Gradio UI, API FastAPI. Out of scope MVP: nhiều người cùng ảnh, video dài, 4 nhãn ASVP còn lại (pleasure/pain/disappointment/breath), mobile/edge yếu. |
| Tech construction | PyTorch 2.x, torchvision, Albumentations, librosa, scikit-learn, matplotlib (vẽ PNG rồi nhúng base64, không CDN), TensorBoard. Chạy local GPU (fallback CPU-only), batch offline. Tải bổ sung bằng `kagglehub` + `huggingface_hub`. |
| Tiêu chí đánh giá | Model level: Accuracy ≥ 0,65 (mặt FER2013 test 7 lớp), ≥ 0,70 (tiếng test 10 lớp, chấp nhận khó hơn vì 3 lớp mới ít mẫu), F1 macro ≥ 0,60, fusion (7 lớp chung) tốt hơn từng nhánh ≥ 3 điểm. Pipeline: latency end-to-end ≤ 200 ms/mẫu trên GPU, `report.html` < 10 MB, mở offline được. Không leakage speaker. |

## Architecture Snapshot (Phase 2 — DESIGN)
- System architecture: late fusion 2 nhánh độc lập với số lớp khác nhau, gặp nhau ở 7 lớp chung khi fusion, chi tiết xem `docs/architecture.md`.
- Modules: `data` (load/augment/split mặt + tiếng, map 3 nguồn) | `models` (face_cnn 7 lớp, audio_cnn 10 lớp, fusion căn lớp) | `training` (trainer riêng từng nhánh, best theo F1 val) | `evaluation` (metric 2 lớp + `visualize_html.py`) | `demo` (script 1 lệnh). Trách nhiệm chi tiết xem `docs/api.md`.
- Folder structure: theo chuẩn skill (`templates/folder-structure.md`), đã có đủ `data/`, `src/`, `models/`, `configs/`, `results/runs/`, `results/demo/`, `docs/`; root chỉ giữ `README.md`.
- Data flow: `raw (fer2013.csv + 21.109 wav) → processed (35.887 PNG 48x48 + 21.109 mel .npy 128x128) → splits (mặt theo Usage, tiếng 70/15/15 theo speaker, seed 42) → train → predictions → results/runs/<ts>/{metrics.json, report.html}` (batch offline, chi tiết xem `docs/data-flow.md`).
- Interfaces / APIs: `FaceDataset → {image [1,48,48], label 0-6}`; `AudioDataset → {mel [1,128,128], label 0-9}` (đọc nhanh qua `audio_*_proc.csv` cột `mel_path`); `FaceCNN → probs[7]`; `AudioCNN → probs[10]`; `fuse(p_face[7], p_audio[10], w) → probs_fused[7]` (cắt 7 lớp đầu + chuẩn hoá lại); `visualize_html.build_report(run_dir) → report.html`. Chi tiết xem `docs/api.md`.
- Milestones: W1: scaffold + Define/Design ✅ | W2: data pipeline mặt + tiếng (đã xong 21.109 WAV + EDA 10 lớp, commit `191e804`) ✅ | W3: model 2 nhánh + train baseline (code xong, sanity xong, chờ train full) ⏳ | W4: late fusion + eval + `report.html` (khung xong) ⏳ | W5: demo 1 lệnh + README + nộp bài ⏳.
- Technical risks: FER2013 mất cân bằng disgust (1,5% train) → weighted loss + augment; 3 lớp tiếng mới ít mẫu (excited chỉ 401) → class weight + SpecAugment; leakage speaker → split theo speaker đã kiểm chứng rỗng; VRAM → train riêng từng nhánh; xem bảng `Risks & Mitigations` bên dưới.

## Active Module: Define + Design — rà soát lần 2, chốt theo dữ liệu thực tế
- **Describe** — Rà soát chéo toàn bộ splits/processed/configs/code rồi viết lại Define 6 mục + Design 7 mục cho khớp dữ liệu thật (21.109 WAV, 246 speaker, tiếng 10 lớp, fusion 7 lớp chung).
- **Goal / purpose** — Mọi tài liệu Define/Design phải nói đúng số liệu đã kiểm chứng, không còn câu chữ cũ (7 lớp tiếng, 8.882 mẫu, RAVDESS-only).
- **Pipeline** — `splits + processed + configs + src → đối chiếu docs/description + architecture + api + progress_status → sửa → pytest + loader check → commit`.
- **How / experiment plan (sub-tasks)**
  - [x] Kiểm tra `audio_files.csv`: 21.109 dòng, phân bố `{0: 2916, 1: 1463, 2: 2643, 3: 3331, 4: 4157, 5: 1778, 6: 2973, 7: 899, 8: 548, 9: 401}`, nguồn `{asvp: 12227, crema: 7442, ravdess: 1440}`, 246 speaker.
  - [x] Kiểm tra split: train 14.457 / val 3.200 / test 3.452, giao speaker train-val/test rỗng, test phân bố đủ 10 lớp.
  - [x] Kiểm tra vật liệu hoá: 35.887 PNG mặt + 21.109 mel `.npy`, `audio_*_proc.csv` khớp 100%, mel thiếu 0 file.
  - [x] Kiểm tra loader thật: mặt `[4,1,48,48]`, tiếng `[4,1,128,128]`; `pytest tests/unit` pass 4/4.
  - [x] Viết lại `docs/description.md` (Define 6 mục + How/Eval), `docs/architecture.md` (số liệu chốt + lý do căn lớp), `docs/api.md` (label 0–9, `mel_path`, output `[B,10]`/`[B,7]`).
  - [x] Cập nhật mục Project Context + Architecture Snapshot trong file này.
  - [ ] Commit Define/Design đợt 2.
- **Status** — Describe ✅ / Generate ✅ / Review ✅ / Run ✅ (số liệu thật + pytest 4/4) / Test ✅ / Refactor ✅ / Commit ⏳
- **Last run** — `pytest tests/unit -q` pass 4/4; `results/eda/report.html` 5,1 MB gallery 10 cảm xúc.
- **DoD checklist** — xem `checklists/dod.md` (workspace skill)

## Module Log
| # | Module | Ngày | Status | Commit | Ghi chú |
|---|--------|------|--------|--------|---------|
| 0 | Scaffold + Define/Design | 2026-11-01 | ✅ | — | folders + docs/progress_status + README + docs/AI_Usage + docs + configs + docs/description.md |
| 1 | data (mặt + tiếng) | 2026-11-01 | ✅ | — | FER 35.887 mẫu + RAVDESS 1.440 WAV 24 speaker, split seed 42, loader thật pass shape |
| 2 | models (face_cnn + audio_cnn + fusion) | 2026-11-01 | ✅ | — | mặt [B,7], tiếng [B,10], fusion [B,7] chuẩn hóa tổng 1, face ~11,17M + audio ~1,21M, pytest 4/4 |
| 3 | training + evaluation + visualization + demo | 2026-11-01 | ✅ code, ⏳ train thật | — | trainer + eval 2 lớp + report.html + demo 1 lệnh, sanity CPU xong |
| 4 | docs bổ sung | 2026-11-01 | ✅ | — | thêm models, training, evaluation, visualization, demo, setup, runbook, decisions, README index trong docs/ |
| 5 | EDA dữ liệu | 2026-11-03 | ✅ | — | script notebooks/2026-11-03_eda.py, plots results/eda/, docs/eda.md |
| 6 | data ASVP-ESD (21.109 WAV, 10 lớp) | 2026-11-09 | ✅ | `191e804` | gộp 3 nguồn, split 246 speaker, đủ 21.109 mel, EDA 10 lớp, report.html 5,1 MB |
| 7 | Define/Design rà soát lần 2 | 2026-11-09 | ⏳ | — | viết lại description + architecture + api + progress_status theo số liệu thật |

## Decisions Log
| Ngày | Quyết định | Lựa chọn khác | Lý do |
|------|------------|---------------|-------|
| 2026-11-01 | 7 lớp mặt theo FER2013, tiếng mở rộng 10 lớp | 4 cảm xúc gọn / 8 cảm xúc | Khớp FER2013 + RAVDESS/CREMA-D/ASVP, đủ khó mà vẫn đạt accuracy khả thi; 3 lớp mới chỉ có ở tiếng. |
| 2026-11-01 | FER2013 (mặt) + RAVDESS/CREMA-D/ASVP-ESD (tiếng) | AffectNet + IEMOCAP | Nhẹ, dễ tải, license nghiên cứu, đủ làm MVP trên GPU lab. |
| 2026-11-01 | Late fusion gộp xác suất, căn về 7 lớp chung | Early fusion / chỉ làm mặt | Hai nhánh độc lập dễ train, dễ ablation, đúng yêu cầu đa mô hình; tiếng 10 lớp bắt buộc phải căn lớp. |
| 2026-11-01 | Local GPU + `report.html` offline | Realtime webcam + micro | Chấm bài không cần mạng, ít rủi ro latency, đúng chuẩn skill. |
| 2026-11-09 | ASVP-ESD lấy bản Kaggle `dejolilandry/asvpesdspeech-nonspeech-emotional-utterances`, mã cảm xúc ở token thứ 3 | Hugging Face `EdwardLin2023/ASVP_ESD` qua `datasets` | Script HF đã deprecated, bản Kaggle đủ 13.964 WAV Audio + Bonus, giữ 12.227 file thuộc 9/13 nhãn, đọc được offline. |
| 2026-11-09 | Loại 4 nhãn ASVP (pleasure/pain/disappointment/breath) | Giữ đủ 13 nhãn | 4 nhãn không ánh xạ được vào thang 10 cảm xúc của đề tài, ít mẫu và khác phân bố NonSpeech. |
| 2026-11-09 | Đọc mel nhanh qua `audio_*_proc.csv` (cột `mel_path`) | Đọc WAV on-the-fly bằng librosa lúc train | Train nhanh hơn, không phụ thuộc librosa lúc train, đã kiểm chứng thiếu 0 file. |

## Risks & Mitigations
| Rủi ro | Khả năng | Tác động | Giảm thiểu | Phát hiện |
|--------|----------|----------|------------|-----------|
| FER2013 mất cân bằng (disgust 1,5% train) | Cao | Trung bình | Weighted cross-entropy + augment mặt, báo F1 macro | Phân bố lớp trong report |
| 3 lớp tiếng mới ít mẫu (excited 401, boredom 548) | Cao | Trung bình | Class weight + SpecAugment mạnh, theo dõi F1 từng lớp | Confusion 10x10 trong report |
| Leakage speaker ở nhánh tiếng | Trung bình | Cao | Split theo speaker seed 42, đã kiểm chứng giao rỗng | So sánh accuracy val/test |
| Audio NonSpeech ASVP khác phân bố Speech | Trung bình | Trung bình | Giữ cột `source`, đánh giá per-source, augment nhiễu | Nghe thử + xem mel mẫu |
| VRAM thiếu khi train 2 nhánh | Trung bình | Trung bình | Train riêng từng nhánh, batch 64/32, mixed precision | Log GPU trong train |

## Open Questions
- [x] Link tải FER2013 và RAVDESS chính thức — đã chốt: FER2013 `data/raw/fer2013.csv`, RAVDESS `data/raw/audio/`, CREMA-D Kaggle `ejlok1/cremad`, ASVP-ESD Kaggle `dejolilandry/asvpesdspeech-nonspeech-emotional-utterances`.
- [ ] Trọng số late fusion để mặc định 0,5/0,5 hay học trên val (`tune_on_val: true` — chốt sau baseline fusion)?
- [ ] Ngưỡng pass cuối: Accuracy fusion ≥ 0,70 có cần điều chỉnh sau baseline 10 lớp không?
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
