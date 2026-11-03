---
name: ai-fusion-workflow
description: Quy trình chuẩn để làm project AI/ML từ Define → Design → Build → Eval + Visualize HTML → Demo → Document. Áp dụng khi người dùng bắt đầu một project AI mới, scaffold thư mục, viết README.md, docs/AI_Usage.md, docs/progress_status.md, sinh report.html trực quan hóa, hoặc khi cần trả lời bộ câu hỏi Define 6 mục trước khi code. Auto-invoke khi thấy project AI/ML hoặc yêu cầu lập kế hoạch.
---

# AI Fusion Workflow

Quy trình chuẩn để làm project AI/ML có trách nhiệm, có tài liệu, và đo lường được.

## Nguyên tắc cốt lõi

1. **Define trước, code sau** — không bao giờ viết dòng code nào khi chưa trả lời được bộ câu hỏi bắt buộc.
2. **MVP trước, hoàn thiện sau** — build module nhỏ nhất chạy được, rồi mới nâng cấp.
3. **Một module một lần, một thay đổi một lần** — tránh mega-commit.
4. **Describe → Generate → Review → Run → Test → Refactor → Commit** — vòng lặp chuẩn.
5. **CONTEXT là vua** — mọi quyết định, lý do, kết quả đều phải được ghi vào `docs/progress_status.md`.
6. **AI là công cụ, người là chủ** — mọi output AI phải được review, sửa, và xác minh trước khi merge.

## Khi nào dùng skill này

- Người dùng mở một project AI/ML mới và muốn scaffold.
- Người dùng yêu cầu lập kế hoạch, định nghĩa yêu cầu, thiết kế kiến trúc.
- Người dùng viết hoặc cập nhật `docs/progress_status.md`, `README.md`, `docs/AI_Usage.md`.
- Người dùng nói: "định nghĩa yêu cầu", "thiết kế hệ thống", "đánh giá model", "làm MVP", "tổ chức project", "visualize", "vẽ biểu đồ", "làm báo cáo html", "làm demo".

## Khi nào KHÔNG dùng

- Project không phải AI/ML (web thuần, CRUD đơn giản).
- Câu hỏi debug lỗi đơn lẻ không liên quan cấu trúc project.
- Đang trong phiên sửa bug nóng, cần fix ngay trong vài phút.

## Quy trình tổng thể (đúng thứ tự file trong `modules/`)

```
Define (description)
        |
        v
Design (architecture + plan)
        |
        v
Build: Data → Model → Training
        |
        v
Eval (model + pipeline) + Visualize HTML (report.html)
        |
        v
Demo (1 lệnh chạy được)
        |
        v
Document (README + docs/AI_Usage + docs/progress_status) — cập nhật song song mọi giai đoạn
```

Áp dụng theo 6 giai đoạn:

### Giai đoạn 1 — DEFINE (đọc `modules/description.md`)
Trả lời bộ câu hỏi bắt buộc (xem `checklists/define-questions.md`). Bắt buộc đủ 6 mục trước khi code:
- **Project name** — tên đầy đủ, slug, tagline một câu.
- **Problem** — bài toán là gì, tại sao quan trọng?
- **Target users** — ai dùng, trong bối cảnh nào, kỳ vọng gì ở output?
- **Input / Output** — dữ liệu vào/ra là gì (kèm shape/định dạng), lấy từ đâu, chia train/val/test thế nào?
- **Core features** — phân loại Must / Should / Could + phạm vi loại trừ.
- **Technology criteria** — chạy ở đâu (Where), khi nào (When), bằng gì (Which), theo tiêu chí chọn nào (phần cứng, hiệu năng, giấy phép, kỹ năng nhóm, deploy)?

Sau khi trả lời, điền vào bảng "Project Context" trong `docs/progress_status.md`. Còn mục nào ghi "TBD" thì liệt kê vào "Open Questions". Chưa đủ 6 mục thì chưa sang Design.

### Giai đoạn 2 — DESIGN
Điền mục "Architecture Snapshot" trong `docs/progress_status.md`:
- **System architecture** — sơ đồ tổng thể.
- **Modules** — danh sách module và trách nhiệm.
- **Folder structure** — cấu trúc thư mục theo chuẩn (xem `templates/folder-structure.md`).
- **Data flow** — dữ liệu đi từ đâu đến đâu.
- **Interfaces / APIs** — đầu vào, đầu ra từng module.
- **Milestones** — mốc thời gian theo tuần hoặc theo module.
- **Technical risks** — rủi ro kỹ thuật và cách giảm thiểu.

### Giai đoạn 3 — BUILD MVP (đọc `modules/data.md` → `modules/model.md` → `modules/training.md`)
Thứ tự build bắt buộc: Data trước, Model sau, Training cuối. Với mỗi module, đọc file chuẩn tương ứng trong `modules/` trước khi code, rồi lặp lại vòng:
1. **Describe** — module làm gì, input/output, tiêu chí thành công (theo `modules/description.md`).
2. **Plan** — dùng `prompts/plan.md` để ra Interfaces + Milestones + Risks, chưa viết code.
3. **Generate** — dùng `prompts/implement.md` để viết code.
4. **Review** — tự review theo `prompts/review.md`.
5. **Run** — chạy thử, ghi log vào `results/runs/<ts>/`.
6. **Test** — unit test cho flow, integration test cho pipeline (xem `prompts/debug.md` nếu fail).
7. **Refactor → Commit** — dọn code, thêm docstring, commit Conventional Commit (một module = một commit).

### Giai đoạn 4 — EVAL + VISUALIZE (đọc `modules/evaluation.md` → `modules/visualization.md`)
Đánh giá xong là visualize ngay, không để số nằm rời rạc trong log:
- **Eval 2 lớp** — model level (accuracy / precision / recall / F1 / mAP / IoU / inference time, tùy bài toán) và full pipeline (latency end-to-end, throughput, resource usage, error handling). Kết quả ghi vào `results/runs/<ts>/metrics.json`.
- **Visualize HTML** — từ `metrics.json` + `plots/` render ra `results/runs/<ts>/report.html` duy nhất, offline 100%, mở bằng double-click (xem `modules/visualization.md`, khung `templates/report.html`, script `src/evaluation/visualize_html.py`). Bắt buộc có: bảng metric pass/fail, đồ thị loss, mẫu tốt/xấu, case tệ nhất, latency pipeline.

### Giai đoạn 5 — DEMO (đọc `modules/demo.md`)
Biến run tốt nhất thành demo 1 lệnh chạy được:
- Input mẫu nhẹ trong `data/processed/demo/`, script `python -m src.demo`, đo latency live, lưu vào `results/demo/<ts>/` kèm `index.html` rút gọn.
- Đồng bộ khối Example Input → Output vào README.

### Giai đoạn 6 — DOCUMENT (song song mọi giai đoạn, chốt ở cuối)
Luôn cập nhật song song, chốt đầy đủ khi demo xong:
- `README.md` — 10 mục chuẩn (xem `templates/readme.md`).
- `docs/AI_Usage.md` — 7 mục chuẩn (xem `templates/ai-usage.md`).
- `docs/progress_status.md` — module log, decisions log, risks.

## Quy tắc làm việc với AI (AI Reference)

### Naming convention
- Model: `{domain}_{architecture}_{version}.{ext}` — ví dụ `face_unet_v1.pt`.
- Dataset: `{purpose}_{date}.{ext}` — ví dụ `train_faces_20261015.zip`.
- Script: `{action}_{target}.py` — ví dụ `train_model.py`, `eval_pipeline.py`.
- Config: `{component}_config.yaml` — ví dụ `model_config.yaml`.
- Run folder: `results/runs/YYYYMMDD-HHMM/`.

### Folder convention
Theo `templates/folder-structure.md`. Tóm tắt:
```
project-root/
├── data/{raw, processed, splits}
├── src/{data, models, training, evaluation}
├── models/{checkpoints, trained, configs}
├── notebooks/
├── tests/
├── docs/
│   ├── architecture.md
│   ├── data-flow.md
│   ├── api.md
│   ├── description.md
│   ├── data.md
│   ├── AI_Usage.md
│   ├── progress_status.md
│   └── CHANGELOG.md
├── configs/
├── results/runs/<ts>/
└── README.md
```

### Discussion rule
Mọi quyết định quan trọng phải ghi vào `docs/progress_status.md` mục "Decisions Log":
```
[YYYY-MM-DD] Decision: <what> | Reasoning: <why> | Alternatives: <what else>
```

### Edition rule
- Version mọi model và dataset.
- Tag release trong git.
- Giữ `docs/CHANGELOG.md` cho model version.
- Ghi rõ breaking change.

## Definition of Done (áp dụng mỗi module)
- [ ] Đúng yêu cầu (đáp án của bộ câu hỏi What/Why/Where-When-Which/How).
- [ ] Code chạy được end-to-end trên môi trường sạch.
- [ ] Test pass (unit + integration).
- [ ] Không có secrets, không commit file lớn.
- [ ] Code đã được self-review.
- [ ] Có thể giải thích được từng dòng cho người khác.
- [ ] Docs đã cập nhật (README, docs/progress_status, docs/AI_Usage).
- [ ] Đã commit theo Conventional Commit.

## Prompt template chuẩn

Có 4 prompt đã soạn sẵn trong thư mục `prompts/`:
1. `plan.md` — lập kế hoạch cho một module.
2. `implement.md` — viết code cho một module.
3. `review.md` — review code AI đã generate.
4. `debug.md` — gỡ lỗi khi test fail.

Mỗi khi bắt đầu một hành động tương ứng, copy file prompt, điền thông tin module cụ thể, rồi gửi cho AI.

## Lớp chuyên môn (Domain expertise)

Khi làm project AI, hãy tự đặt mình vào các vai trò:
- **Product Owner** — định nghĩa yêu cầu, ưu tiên tính năng.
- **Domain Expert** — cung cấp kiến thức chuyên ngành, validate output.
- **AI Architect** — thiết kế kiến trúc, chọn tech stack.
- **AI Engineer** — code model, build pipeline, tối ưu.
- **AI Operator** — deploy, monitor, maintain.
- **Security** — đảm bảo privacy, model security, compliance.

AI agent hoạt động theo 3 lớp:
```
Prompting (mô tả + luật)
        |
        v
Skill / Behavioral (ra quyết định)
        |
        v
Brain / Knowledge base (context dài hạn)
```

## CONTEXT — tại sao quan trọng

CONTEXT là thứ duy nhất khiến AI trả lời đúng. Khi mất context, AI sẽ:
- Đoán sai ý bạn.
- Lặp lại quyết định đã bác bỏ.
- Vi phạm convention đã thống nhất.

Cách giữ context:
- `docs/progress_status.md` luôn được cập nhật sau mỗi module.
- Mỗi prompt phải chứa: mục tiêu module, ràng buộc, file liên quan, kết quả mong đợi.
- Không để AI "đoán" — ghi rõ input/output shape, metric, threshold.

## Tài nguyên trong skill (sắp theo luồng Define → Design → Build → Eval → Demo)

**A. Checklist — hỏi và nghiệm thu:**
- `checklists/define-questions.md` — bộ câu hỏi Define đủ 6 mục (project name, problem, target users, input/output, core features, technology criteria).
- `checklists/dod.md` — Definition of Done chi tiết cho mọi module.

**B. Modules — chuẩn từng khối công việc (đọc trước khi code module đó):**
1. `modules/description.md` — chuẩn viết mô tả project / module (dùng ở Define).
2. `modules/data.md` — chuẩn data pipeline: raw → processed → splits, augment, chống leakage.
3. `modules/model.md` — chuẩn chọn kiến trúc, baseline, loss, version model.
4. `modules/training.md` — chuẩn vòng train: config, checkpoint, resume, early stopping.
5. `modules/evaluation.md` — chuẩn đánh giá 2 lớp model level + full pipeline.
6. `modules/visualization.md` — chuẩn trực quan hóa bằng một file HTML offline (`report.html`).
7. `modules/demo.md` — chuẩn script demo 1 lệnh + kịch bản trình diễn.

**C. Prompts — template làm việc với AI:**
- `prompts/plan.md` — lập kế hoạch cho một module (đã gồm Interfaces + Milestones + Risks).
- `prompts/implement.md` — viết code cho một module.
- `prompts/review.md` — review code AI đã generate.
- `prompts/debug.md` — gỡ lỗi khi test fail.

**D. Templates — khung file copy-paste:**
- `templates/folder-structure.md` — cấu trúc thư mục chuẩn.
- `templates/progress-status.md` — khung `docs/progress_status.md` đủ Define 6 mục + Design 7 mục.
- `templates/readme.md` — README 10 mục.
- `templates/ai-usage.md` — AI_Usage 7 mục.
- `templates/report.html` — khung HTML cho `results/runs/<ts>/report.html` (dùng với `modules/visualization.md`).

## Cách dùng nhanh

Khi user bảo "bắt đầu project AI mới":
1. Đọc `modules/description.md` + `checklists/define-questions.md` và hỏi user các câu hỏi chưa rõ.
2. Điền `docs/progress_status.md` mục Project Context.
3. Đọc `templates/folder-structure.md` và scaffold thư mục.
4. Đọc `templates/readme.md`, `templates/ai-usage.md`, `templates/progress-status.md` để user có file mẫu.
5. Sau đó đi đúng luồng: Design → Build (Data → Model → Training) → Eval + Visualize HTML → Demo → Document.

Khi user bảo "code module X":
1. Đọc file chuẩn trong `modules/` tương ứng (`data.md` / `model.md` / `training.md` / `evaluation.md` / `visualization.md` / `demo.md`), rồi đọc `prompts/plan.md` để ra Interfaces + Milestones trước, sau đó dùng `prompts/implement.md` để code.
2. Sau khi nhận code, dùng `prompts/review.md` để review.
3. Chạy và commit theo vòng lặp.

Khi user bảo "đánh giá model" / "đánh giá pipeline":
1. Đọc `modules/evaluation.md`, tách rõ model level và full pipeline.
2. Ghi kết quả vào `results/runs/<ts>/metrics.json`.
3. Cập nhật `docs/progress_status.md` mục "Module Log".

Khi user bảo "visualize" / "vẽ biểu đồ" / "làm báo cáo html":
1. Đọc `modules/visualization.md`, dùng khung `templates/report.html`.
2. Viết `src/evaluation/visualize_html.py` render từ `metrics.json` + `plots/` ra `results/runs/<ts>/report.html` offline một file duy nhất.
3. Kiểm tra double-click mở được trên máy khác, số khớp `metrics.json`.

Khi user bảo "làm demo":
1. Đọc `modules/demo.md`, chuẩn bị input mẫu trong `data/processed/demo/`.
2. Viết script 1 lệnh chạy được, đo latency live, lưu output vào `results/demo/<ts>/`.
3. Đồng bộ khối Example Input → Output vào README.