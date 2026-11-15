# Hệ thống nhận diện cảm xúc con người dựa trên biểu cảm khuôn mặt và tín hiệu âm thanh sử dụng Deep Learning

> Single source of truth cho **đang làm gì, tại sao, và đã làm được gì**.
> Cập nhật lần cuối: 2026-11-15 (đồng bộ docs theo code: xoá code EVAL/VISUALIZE/DEMO để xây lại)

## Project Context (Phase 1 — DEFINE)
| Trường | Giá trị |
|--------|---------|
| Tên đầy đủ | Nghiên cứu và xây dựng hệ thống nhận diện cảm xúc con người dựa trên biểu cảm khuôn mặt và tín hiệu âm thanh sử dụng Deep Learning |
| Slug | `emotion-fusion` |
| Tagline | Nhận diện 10 cảm xúc từ ảnh khuôn mặt và giọng nói bằng Deep Learning với fusion muộn; cả 2 nhánh dùng chung thang 10 lớp, báo cáo bằng một file HTML offline. |
| Bài toán | Phân loại (classification) cảm xúc từ 2 modalities. Cả 2 nhánh dùng thang 10 lớp: angry (0), disgust (1), fear (2), happy (3), sad (4), surprise (5), neutral (6), contempt (7), boredom (8), excited (9). Nhánh mặt: 7 lớp gốc theo FER2013 + 3 lớp mới bổ sung từ nguồn ảnh bên ngoài (AffectNet contempt, Student Engagement boredom, AffectNet happy làm proxy excited). Nhánh tiếng: 7 lớp gốc từ RAVDESS/CREMA-D + 3 lớp mới từ ASVP-ESD. Fusion gộp trực tiếp 10 lớp vì 2 phân bố đã cùng thang, không cần cắt lớp. Quan trọng vì cảm xúc là tín hiệu cốt lõi trong tương tác người–máy, hỗ trợ giáo dục, tư vấn và chăm sóc khách hàng. |
| Người dùng mục tiêu | Sinh viên thực hiện đồ án và giảng viên chấm bài; nhà nghiên cứu AI cần baseline đa mô hình gọn nhẹ. Bối cảnh: chạy local GPU trong phòng lab (fallback CPU-only), xem kết quả qua `report.html` và `results/eda/report.html`, demo 1 lệnh. Kỳ vọng output: nhãn cảm xúc + phân bố xác suất (mặt 10, tiếng 10, fusion 10) + độ trễ suy luận. |
| Input | Nhánh mặt: ảnh xám 48x48, nguồn FER2013 (`data/raw/fer2013.csv`, 35.887 mẫu: train 28.709 / val 3.589 / test 3.589 theo cột `Usage` gốc) + 3.904 ảnh extra cho lớp 7–9 (contempt 2.946, boredom 358, excited 600; `Usage=extra`), tổng 39.791 mẫu: train 31.441 / val 4.175 / test 4.175. Nhánh tiếng: file WAV 16 kHz mono 3–5 giây, chuẩn hoá về 4 giây thành Mel 128x128; nguồn gộp 3 bộ tổng 21.109 WAV, 246 speaker (RAVDESS 1.440 + CREMA-D 7.442 + ASVP-ESD 12.227 file dùng được), split theo speaker seed 42: train 14.457 (172 speaker) / val 3.200 (36) / test 3.452 (38), giao nhau rỗng. Demo dùng ảnh PNG + WAV mẫu trong `data/processed/demo/`. |
| Output | JSON: `{ emotion, confidence, probs_face[10], probs_audio[10], probs_fused[10], inference_time_ms }` + `report.html` trong `results/runs/<ts>/` và `index.html` rút gọn trong `results/demo/<ts>/`. |
| Tính năng cốt lõi | Must: data pipeline mặt (48x48, 39.791 mẫu 10 lớp) + tiếng (mel 128x128, 21.109 `.npy`), Face-CNN 10 lớp + Audio-CNN 10 lớp + late fusion 10 lớp, eval 2 lớp, `report.html` offline, demo 1 lệnh. Should: confusion matrix per-class (mặt 10x10, tiếng 10x10, fusion 10x10) + 5 case tệ nhất, webcam/micro realtime, `index.html` demo. Could: export ONNX, Gradio UI, API FastAPI. Out of scope MVP: nhiều người cùng ảnh, video dài, 4 nhãn ASVP còn lại (pleasure/pain/disappointment/breath), mobile/edge yếu. |
| Tech construction | PyTorch 2.x, torchvision, Albumentations, librosa, scikit-learn, matplotlib (vẽ PNG rồi nhúng base64, không CDN), TensorBoard. Chạy local GPU (fallback CPU-only), batch offline. Tải bổ sung bằng `kagglehub` + `huggingface_hub`. |
| Tiêu chí đánh giá | Model level: Accuracy ≥ 0,65 (mặt FER2013 test 10 lớp, chấp nhận khó hơn vì 3 lớp mới ít mẫu), ≥ 0,70 (tiếng test 10 lớp), F1 macro ≥ 0,60, fusion (10 lớp) tốt hơn từng nhánh ≥ 3 điểm. Pipeline: latency end-to-end ≤ 200 ms/mẫu trên GPU, `report.html` < 10 MB, mở offline được. Không leakage speaker. |

## Architecture Snapshot (Phase 2 — DESIGN)
- System architecture: late fusion 2 nhánh độc lập **cùng thang 10 lớp**, gặp nhau khi fusion (không cần cắt lớp), chi tiết xem `docs/architecture.md`.
- Modules: `data` (load/augment/split mặt + tiếng, map 3 nguồn tiếng + 3 nguồn ảnh extra) | `models` (face_cnn 10 lớp, audio_cnn 10 lớp, fusion 10 lớp) | `training` (trainer riêng từng nhánh, best theo F1 val) | `evaluation` ⏳ *chưa triển khai* | `demo` ⏳ *chưa triển khai*. Trách nhiệm chi tiết xem `docs/api.md`.
- Folder structure: theo chuẩn skill (`templates/folder-structure.md`), đã có đủ `data/`, `src/`, `models/`, `configs/`, `results/runs/`, `results/demo/`, `docs/`; root chỉ giữ `README.md`.
- Data flow: `raw (fer2013.csv + 3.904 ảnh extra + 21.109 wav) → processed (39.791 ảnh 48x48 + 21.109 mel .npy 128x128) → splits (mặt theo Usage + extra 70/15/15, tiếng theo speaker seed 42) → train → predictions → results/runs/<ts>/{metrics.json, report.html}` (batch offline, chi tiết xem `docs/data-flow.md`).
- Interfaces / APIs: `FaceDataset → {image [1,48,48], label 0-9}`; `AudioDataset → {mel [1,128,128], label 0-9}` (đọc nhanh qua `audio_*_proc.csv` cột `mel_path`); `FaceCNN → probs[10]`; `AudioCNN → probs[10]`; `fuse(p_face[10], p_audio[10], w) → probs_fused[10]` (căn về `min` số lớp + chuẩn hoá lại); `visualize_html.build_report(run_dir) → report.html` ⏳ *đặc tả*. Chi tiết xem `docs/api.md`.
- Milestones: W1: scaffold + Define/Design ✅ | W2: data pipeline mặt + tiếng (21.109 WAV + ảnh extra, commit `191e804`) ✅ | W3: model 2 nhánh ✅ + train baseline ⏳ (chỉ sanity 1–2 epoch, chờ train full) | W3.5: hyperparameter tuning (3–4 model × 6 siêu tham số × 6–7 giá trị, 160 trial/nhánh) ⏳ | W4: late fusion + eval + `report.html` ⏳ | W5: demo 1 lệnh + README + nộp bài ⏳.
- Technical risks: FER2013 mất cân bằng disgust (1,5% train) → weighted loss + augment; 3 lớp mới ít mẫu (mặt boredom 358, tiếng excited 401) → class weight + SpecAugment; ảnh extra khác domain FER2013 (nguồn khác, chất lượng khác) → có thể ảnh hưởng accuracy 7 lớp gốc; leakage speaker → split theo speaker đã kiểm chứng rỗng; VRAM → train riêng từng nhánh; xem bảng `Risks & Mitigations` bên dưới.

## Active Module: Triển khai EVAL + VISUALIZE + DEMO
- **Describe** — Xây dựng 3 module cuối (EVAL, VISUALIZE, DEMO) theo đặc tả trong `docs/evaluation.md`, `docs/visualization.md`, `docs/demo.md`.
- **Goal / purpose** — Đáp ứng đúng Define (10 lớp cả 2 nhánh, eval 2 lớp có fusion, report.html đủ 6 nhóm nội dung bắt buộc, demo 1 lệnh).
- **Pipeline** `train full → eval metrics → render report.html → demo CLI → cập nhật docs`.
- **How / experiment plan (sub-tasks)** — chi tiết ở `docs/plan_remaining.md`
  - [ ] **Bước 1** — Train + tìm model tối ưu: 1.1 sanity → 1.2 train baseline → 1.3 khai báo tuning space + mở rộng `src/models/` → 1.4 viết `src/training/tuner.py` + lệnh `tune` → 1.5 quét 160 trial/nhánh rồi train full lại → 1.6 lập bảng so sánh.
  - [ ] **Bước 2** — EVAL: 2.1 `src/evaluation/metrics.py` → 2.2 `src/evaluation/__main__.py` → 2.3 ghép cặp face/audio cho fusion.
  - [ ] **Bước 3** — VISUALIZE: 3.1 `src/evaluation/visualize.py` → 3.2 `src/evaluation/visualize_html.py`.
  - [ ] **Bước 4** — DEMO: 4.1 input mẫu (`src/data/make_demo.py`) → 4.2 `src/demo/__main__.py`.
  - [ ] File của bước chưa làm phải trống/placeholder — kiểm tra bằng `python scripts/check_pending_modules.py` (test: `tests/unit/test_pending_modules.py`).
  - [ ] 5 file code EVAL/VISUALIZE/DEMO đã giữ sẵn rỗng (0 byte) để thành viên mở ra viết: `src/evaluation/{__main__,metrics,visualize,visualize_html}.py` + `src/demo/__main__.py`.
- **Status** — Describe ✅ / Generate ⏳ / Review ⏳ / Run ⏳ / Test ⏳ / Refactor ⏳ / Commit ⏳
- **Ghi chú** — Run cũ `results/runs/20261109-1704/` là model 7 lớp, không có fusion, report.html chỉ 2,7 KB → không được trích dẫn.
- **DoD checklist** — xem `checklists/dod.md` (workspace skill)

## Module Log
| # | Module | Ngày | Status | Commit | Ghi chú |
|---|--------|------|--------|--------|---------|
| 0 | Scaffold + Define/Design | 2026-11-01 | ✅ | — | folders + docs/progress_status + README + docs/AI_Usage + docs + configs + docs/description.md |
| 1 | data (mặt + tiếng) | 2026-11-01 | ✅ | — | FER 35.887 mẫu + RAVDESS 1.440 WAV 24 speaker, split seed 42, loader thật pass shape |
| 2 | models (face_cnn + audio_cnn + fusion) | 2026-11-01 | ✅ | — | mặt [B,10], tiếng [B,10], fusion [B,10] chuẩn hóa tổng 1, face ~11,17M + audio ~1,21M, pytest 7/7 (gồm forward 2 backbone) |
| 3 | training + evaluation + visualization + demo | 2026-11-01 | ⏳ | — | trainer còn nguyên; eval + visualize + demo chưa triển khai; sanity CPU xong |
| 4 | docs bổ sung | 2026-11-01 | ✅ | — | thêm models, training, evaluation, visualization, demo, setup, runbook, decisions, README index trong docs/ |
| 5 | EDA dữ liệu | 2026-11-01 | ✅ | — | script notebooks/2026-11-03_eda.py, plots results/eda/, docs/eda.md |
| 6 | data ASVP-ESD (21.109 WAV, 10 lớp) | 2026-11-09 | ✅ | `191e804` | gộp 3 nguồn, split 246 speaker, đủ 21.109 mel, EDA 10 lớp, report.html 5,1 MB |
| 7 | Define/Design rà soát lần 2 | 2026-11-09 | ✅ | `e75ca1b` | viết lại description + architecture + api + progress_status theo số liệu thật |
| 8 | data ảnh extra nhánh mặt (3.904 ảnh, 10 lớp) | 2026-11-15 | ✅ | — | `src/data/add_face_extra.py`: contempt 2.946 (AffectNet) + boredom 358 (Student Engagement) + excited 600 (proxy AffectNet happy); tổng mặt 39.791 mẫu 10 lớp |
| 9 | Define/Design rà soát lần 3 + đồng bộ docs | 2026-11-15 | ✅ | — | sửa code (fusion/demo/eval suy num_classes) + test 7/7 + viết lại 10 file docs + README theo 10 lớp cả 2 nhánh |
| 10 | Triển khai EVAL + VISUALIZE + DEMO | 2026-11-15 | ⏳ | — | xác định đặc tả thiết kế cho 3 module cuối (10 lớp cả 2 nhánh, eval có fusion, report.html đủ 6 nhóm nội dung, demo 1 lệnh); plan phân công `docs/plan_remaining.md` |
| 11 | Plan công việc còn lại (giao thành viên) | 2026-11-15 | ✅ | — | `docs/plan_remaining.md`: 13 bước A1–E3, mỗi bước ghi rõ làm file gì · chức năng file · có thể làm thêm sau; kèm phân công 4 người, timeline 4 ngày, DoD |
| 12 | Đồng bộ docs theo trạng thái triển khai | 2026-11-15 | ⏳ | — | thống nhất cách ghi trạng thái "chưa triển khai" cho eval/visualize/demo; viết lại plan theo từng bước cụ thể; giữ sẵn 5 file code rỗng (0 byte) cho thành viên; thêm cơ chế canh gác `scripts/check_pending_modules.py` + `tests/unit/test_pending_modules.py`; pytest 15/15 pass |
| 13 | Bổ sung khối T — hyperparameter tuning | 2026-11-15 | ⏳ | — | thêm vào plan: mỗi nhánh 4 model variants × 6 siêu tham số (mỗi tham số 6–7 giá trị) × 40 trial = 160 trial/nhánh, random search seed 42, tune trên val; thêm `configs/tuning_space.yaml` + `src/training/tuner.py`; mở rộng `src/models/` cho nhiều kiến trúc |
| 14 | Viết lại plan theo 4 bước | 2026-11-15 | ⏳ | — | `docs/plan_remaining.md` rút từ 756 xuống ~460 dòng; gom thành 4 bước (1 Train+tuning, 2 Eval, 3 Visualize, 4 Demo), mỗi bước giữ 3 ý: làm file gì · chức năng · làm thêm sau; thêm bảng files cần viết/sửa |

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
| 2026-11-15 | **Mở rộng nhánh mặt lên 10 lớp** (thêm contempt/boredom/excited từ nguồn ảnh ngoài) | Giữ nhánh mặt 7 lớp, chỉ tiếng 10 lớp | Đồ án đa mô hình cần cả 2 nhánh cùng thang để fusion không phải cắt bỏ thông tin. Nguồn ảnh: AffectNet contempt (crop 96x96), Student Engagement boredom (center-crop), AffectNet happy làm proxy excited. |
| 2026-11-15 | Fusion gộp trực tiếp 10 lớp, không cắt về 7 lớp chung | Giữ `_align_probs` cắt về 7 | Cả 2 nhánh đã cùng thang 10 lớp và cùng thứ tự label → căn theo `min` là giữ nguyên 10, không mất lớp nào. |
| 2026-11-15 | Suy `num_classes` từ shape checkpoint thay vì hardcode | Hardcode 10 ở mọi nơi | Tương thích checkpoint 7 lớp cũ, tránh crash `load_state_dict` khi đổi số lớp. Áp dụng cho `demo/` và `evaluation/`. |
| 2026-11-15 | Bổ sung test forward thật cho `FaceCNN`/`AudioCNN` | Chỉ test fusion | Trước đó "pytest 4/4" không hề test backbone — chỉ test hàm fusion. Nay 7/7 gồm forward 2 backbone. |
| 2026-11-15 | **Xây lại EVAL + VISUALIZE + DEMO theo đặc tả thiết kế**, triển khai mới cho khớp thang 10 lớp | Sửa lại code cũ cho khớp 10 lớp | Code cũ chỉ hỗ trợ 7 lớp, report.html thiếu nội dung bắt buộc, chưa có đánh giá fusion. Triển khai lại theo đặc tả trong docs để đúng chuẩn. |
| 2026-11-15 | **Random search** cho hyperparameter tuning thay vì grid search đầy đủ | Grid search 6^6 ≈ 46.656 tổ hợp; Optuna | 46.656 lần train không khả thi với ngân sách GPU lab. Random search 160 trial/nhánh cho kết quả tốt gần bằng grid search với ít thử nghiệm hơn nhiều (theo nghiên cứu của Bergstra & Bengio). Nếu có thời gian, nâng lên Optuna. |
| 2026-11-15 | **Tune trên val, test chỉ dùng 1 lần cuối** | Dùng test để chọn siêu tham số | Chọn cấu hình trên test là rò rỉ test set → số liệu báo cáo không đáng tin. Giữ test kín tới bước eval cuối. |
| 2026-11-15 | **4 model variants mỗi nhánh** (resnet18 / resnet18-nopretrain / mobilenetv2 / small_cnn; cnn 3-4-5 blocks) | Chỉ 1 kiến trúc mỗi nhánh | Yêu cầu tìm model tối ưu; có nhiều kiến trúc mới biết cái nào hợp dữ liệu 10 lớp. MobileNetV2 nhẹ hơn → nếu accuracy tương đương thì đạt latency ≤ 200 ms. |

## Risks & Mitigations
| Rủi ro | Khả năng | Tác động | Giảm thiểu | Phát hiện |
|--------|----------|----------|------------|-----------|
| FER2013 mất cân bằng (disgust 1,5% train) | Cao | Trung bình | Weighted cross-entropy + augment mặt, báo F1 macro | Phân bố lớp trong report |
| 3 lớp tiếng mới ít mẫu (excited 401, boredom 548) | Cao | Trung bình | Class weight + SpecAugment mạnh, theo dõi F1 từng lớp | Confusion 10x10 trong report |
| 3 lớp mặt mới rất ít mẫu (boredom 358, excited 600) | Cao | Cao | Class weight mạnh, đánh giá riêng F1 từng lớp mới | Confusion 10x10 nhánh mặt |
| Ảnh extra khác domain FER2013 (nguồn, chất lượng, crop khác) | Cao | Cao | Augment mạnh, kiểm tra confusion 7 lớp gốc không tụt, cân nhắc domain adaptation | So sánh accuracy 7 lớp gốc trước/sau khi thêm extra |
| `excited` ảnh mặt là proxy từ `happy` | Cao | Trung bình | Ghi rõ giới hạn trong báo cáo, không claim là ground truth thật | Kiểm tra trực quan trong `results/eda/real_contempt/` |
| Leakage speaker ở nhánh tiếng | Trung bình | Cao | Split theo speaker seed 42, đã kiểm chứng giao rỗng | So sánh accuracy val/test |
| Audio NonSpeech ASVP khác phân bố Speech | Trung bình | Trung bình | Giữ cột `source`, đánh giá per-source, augment nhiễu | Nghe thử + xem mel mẫu |
| VRAM thiếu khi train 2 nhánh | Trung bình | Trung bình | Train riêng từng nhánh, batch 64/32, mixed precision | Log GPU trong train |

## Open Questions
- [x] Link tải FER2013 và RAVDESS chính thức — đã chốt: FER2013 `data/raw/fer2013.csv`, RAVDESS `data/raw/audio/`, CREMA-D Kaggle `ejlok1/cremad`, ASVP-ESD Kaggle `dejolilandry/asvpesdspeech-nonspeech-emotional-utterances`.
- [ ] Trọng số late fusion để mặc định 0,5/0,5 hay học trên val (`tune_on_val: true` — chốt sau baseline fusion)?
- [ ] Ngưỡng pass cuối: Accuracy mặt 10 lớp ≥ 0,65 có cần điều chỉnh sau khi thêm 3 lớp mới ít mẫu không?
- [ ] Có cần nhánh realtime webcam/micro ở milestone W5 hay để Could?
- [ ] Nguồn ảnh `excited` là proxy từ `happy` (AffectNet) — có cần thu thập ảnh excited thật để tăng độ tin cậy không?
- [ ] Ảnh extra có làm giảm accuracy 7 lớp FER2013 gốc không — cần ablation so sánh.
- [ ] Viết lại EVAL (2 lớp, có fusion) + VISUALIZE (report.html đủ 6 nhóm nội dung) + DEMO (1 lệnh) — thứ tự ưu tiên: EVAL → VISUALIZE → DEMO.
- [ ] Train full 2 nhánh (30 epoch) trước khi eval — checkpoint hiện chỉ sanity 1–2 epoch.
- [ ] Xoá hay giữ output cũ `results/runs/20261109-1704/` (model 7 lớp) và `results/demo/20261109-1706/` (probs 7 phần tử)?

## 📋 Plan công việc còn lại (giao cho thành viên)

> Chi tiết từng bước (làm file gì · chức năng · làm thêm sau được gì): **`docs/plan_remaining.md`**

| Bước | Làm gì | Kết quả |
|------|--------|---------|
| **1** | Train baseline + tìm model tối ưu (4 variants × 6 siêu tham số × 6–7 giá trị, 160 trial/nhánh) | `best.pt` mặt + tiếng |
| **2** | Đánh giá (EVAL) — `metrics.py` + `__main__.py` | `metrics.json` có 3 nhánh |
| **3** | Vẽ báo cáo (VISUALIZE) — `visualize.py` + `visualize_html.py` | `report.html` offline |
| **4** | Demo 1 lệnh (DEMO) — `demo/__main__.py` + input mẫu | `result.json` + `index.html` |

Thứ tự bắt buộc: **1 → 2 → 3 → 4**. Bước 2, 3, 4 chia được cho 3 người làm song song sau khi bước 1 xong.


## Definition of Done (áp dụng mỗi module)
- [x] Đúng yêu cầu — đặc tả thiết kế cho EVAL/VISUALIZE/DEMO khớp thang 10 lớp
- [x] Code chạy end-to-end — `import src.models / src.data / src.training` OK
- [x] Tests pass — `pytest tests/unit` 15/15 (gồm 8 test kiểm tra trạng thái module chưa triển khai)
- [x] Không có secrets / không có artifact lớn
- [x] Code đã self-review — `scripts/check_pending_modules.py` exit 0
- [x] Có thể giải thích được — đặc tả thiết kế đầy đủ trong `docs/evaluation.md`, `docs/visualization.md`, `docs/demo.md`
- [x] Docs đã cập nhật — thống nhất trạng thái "chưa triển khai" cho eval/visualize/demo
- [ ] Đã committed (Conventional Commit)
