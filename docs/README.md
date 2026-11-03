# Mục lục tài liệu Emotion Fusion

> Mọi tài liệu project đều nằm ở đây. Root chỉ giữ `README.md`.

## Define (hiểu bài toán)
- `description.md` — mô tả project: tên, problem, users, input/output, Must/Should/Could, success criteria.
- `progress_status.md` — single source of truth: Project Context (Define 6 mục) + Architecture Snapshot (Design 7 mục) + Module Log.

## Design (hiểu kiến trúc)
- `architecture.md` — sơ đồ late fusion 2 nhánh.
- `data-flow.md` — luồng raw → processed → splits → train → report.
- `api.md` — API nội bộ giữa các module.
- `models.md` — model card chi tiết Face-CNN + Audio-CNN + fusion.
- `data.md` — quy ước thư mục dữ liệu.
- `eda.md` — phân tích khám phá dữ liệu FER2013 + RAVDESS.
- `decisions.md` — log quyết định kỹ thuật.

## Vận hành (chạy project)
- `setup.md` — cài đặt môi trường từ máy sạch.
- `runbook.md` — sổ vận hành end-to-end: split → sanity → train → eval → demo.
- `training.md` — quy trình huấn luyện 2 nhánh.
- `evaluation.md` — đánh giá 2 lớp model + pipeline.
- `visualization.md` — render `report.html` offline.
- `demo.md` — demo 1 lệnh sinh JSON + `index.html`.

## Quản trị
- `AI_Usage.md` — hồ sơ sử dụng AI trung thực.
- `CHANGELOG.md` — lịch sử thay đổi.
