# progress_status.md — khung đầy đủ

Đây là file **single source of truth** cho project: đang làm gì, tại sao, đã làm được gì, còn rủi ro gì.

```markdown
# <Tên đầy đủ của Project>

> Single source of truth cho **đang làm gì, tại sao, và đã làm được gì**.
> Cập nhật lần cuối: <YYYY-MM-DD HH:MM>

## Project Context (Phase 1 — DEFINE)
| Trường | Giá trị |
|--------|---------|
| Tên đầy đủ | <TBD> |
| Slug | <lab1> |
| Bài toán | <TBD> |
| Người dùng mục tiêu | <TBD> |
| Input | <TBD> |
| Output | <TBD> |
| Tính năng cốt lõi | Must: …  Should: …  Could: … |
| Tech construction | <TBD> |
| Tiêu chí đánh giá | <TBD> |

## Architecture Snapshot (Phase 2 — DESIGN)
- System architecture: <pattern + mô tả 1-2 câu, chi tiết xem `docs/architecture.md`>
- Modules: <danh sách module + trách nhiệm mỗi module, ví dụ `data: load/augment | models: kiến trúc | training: vòng train | evaluation: metric`>
- Folder structure: <theo chuẩn skill, chi tiết xem `templates/folder-structure.md`>
- Data flow: `raw → processed → splits → train → predictions → results` (kèm shape, tần suất real-time/batch, chi tiết xem `docs/data-flow.md`)
- Interfaces / APIs: <input/output từng module kèm shape/định dạng, chi tiết xem `docs/api.md`>
- Milestones: <mốc theo tuần hoặc module, ví dụ `W1: scaffold | W2: data | W3: model + train | W4: eval + docs`>
- Technical risks: <tóm tắt 2-3 rủi ro lớn nhất, chi tiết xem bảng `Risks & Mitigations` bên dưới>

## Active Module: <tên module>
- **Describe** — <module làm gì, input/output, tiêu chí thành công>
- **Goal / purpose** — <mục tiêu kinh doanh / kỹ thuật>
- **Pipeline** — <sơ đồ luồng>
- **How / experiment plan (sub-tasks)**
  - [ ] <sub-task 1>
  - [ ] <sub-task 2>
  - [ ] <sub-task 3>
- **Status** — Describe ⏳ / Generate ⏳ / Review ⏳ / Run ⏳ / Test ⏳ / Refactor ⏳ / Commit ⏳
- **Last run** — `results/runs/<timestamp>/`
- **DoD checklist** — xem `checklists/dod.md` (workspace skill)

## Module Log
| # | Module | Ngày | Status | Commit | Ghi chú |
|---|--------|------|--------|--------|---------|
| 0 | Scaffold | <today> | ✅ | — | folders + templates in place |
| 1 | <module> | <date> | ⏳ | <sha> | <ghi chú> |

## Decisions Log
| Ngày | Quyết định | Lựa chọn khác | Lý do |
|------|------------|---------------|-------|
|      |            |               |       |

## Risks & Mitigations
| Rủi ro | Khả năng | Tác động | Giảm thiểu | Phát hiện |
|--------|----------|----------|------------|-----------|
|        |          |          |          |         |

## Open Questions
- [ ] Tên project + bài toán
- [ ] Input/output shape
- [ ] Tính năng cốt lõi (Must/Should/Could)
- [ ] Tech stack + lý do
- [ ] Metric đánh giá + target

## Definition of Done (áp dụng mỗi module)
- [ ] Đúng yêu cầu
- [ ] Code chạy end-to-end
- [ ] Tests pass
- [ ] Không có secrets / không có artifact lớn
- [ ] Code đã self-review
- [ ] Có thể giải thích được
- [ ] Docs đã cập nhật
- [ ] Đã committed (Conventional Commit)
```

## Cách dùng

- **Mỗi module** — tạo mục "Active Module" mới, điền đầy đủ trước khi code.
- **Mỗi commit** — thêm dòng vào "Module Log" (ngày + commit hash + ghi chú).
- **Mỗi quyết định quan trọng** — thêm dòng vào "Decisions Log".
- **Mỗi rủi ro mới phát hiện** — thêm dòng vào "Risks & Mitigations".
- **Mỗi câu hỏi chưa giải quyết** — thêm dòng vào "Open Questions".

Khi tất cả "Open Questions" được tick ✅ và "Active Module" có Status = ✅ Commit, thì project sẵn sàng cho MVP hoàn chỉnh.