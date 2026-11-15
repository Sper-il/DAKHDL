# Definition of Done — chi tiết

Áp dụng cho **mọi module** và **mọi commit**. Tick đủ thì mới merge.

## Đúng yêu cầu
- [ ] Module giải quyết đúng bài toán đã định nghĩa trong bộ câu hỏi What/Why/Where-When-Which/How.
- [ ] Input/output khớp với mục "Interfaces / APIs" trong `progress_status.md`.
- [ ] Tính năng nằm đúng trong phân loại Must / Should / Could.

## Code chạy được
- [ ] Chạy được end-to-end trên máy sạch (clone fresh → cài dependency → chạy).
- [ ] Không phụ thuộc file local không có trong repo (trừ data đã ghi rõ).
- [ ] Có log rõ ràng khi chạy (info / warning / error).

## Test pass
- [ ] Unit test cho function / class quan trọng.
- [ ] Integration test cho luồng chính (input → output).
- [ ] Smoke test chạy được trong < 1 phút.
- [ ] Tất cả test xanh, không có warning nghiêm trọng.

## Không có secrets
- [ ] Không commit API key, token, password.
- [ ] Không commit dataset lớn (> 50 MB) — dùng `.gitignore` hoặc DVC / LFS.
- [ ] Không commit model đã train nặng (> 100 MB) — trừ khi dùng LFS.
- [ ] Không commit file output tạm (`__pycache__`, `.pyc`, `*.log`, `*.tmp`).

## Code đã review
- [ ] Đã đọc lại toàn bộ code AI generate.
- [ ] Đã sửa các chỗ sai convention.
- [ ] Đã thêm docstring cho function / class public.
- [ ] Đã thêm type hint cho Python (nếu ngôn ngữ cho phép).
- [ ] Đã xóa code chết, print debug thừa.

## Giải thích được
- [ ] Có thể giải thích từng quyết định thiết kế cho người khác hiểu.
- [ ] Có thể chỉ ra tại sao chọn model / hyperparameter này.
- [ ] Không có đoạn code "magic" không giải thích được.

## Docs đã cập nhật
> **Điều kiện bắt buộc.** Không tick mục này thì module chưa xong — chưa được đánh dấu ✅ trong Module Log và chưa được commit. Xem mục "Quy tắc bắt buộc — Làm xong module = đã cập nhật docs" trong `SKILL.md` (Giai đoạn 3).

- [ ] **Đã quét thay đổi** — liệt kê mọi thay đổi code vừa làm, map sang bảng "Thay đổi trong code → Docs bắt buộc phải sửa".
- [ ] **Đã sửa docs theo số liệu thật** — đọc lại từ `configs/`, `splits/`, `processed/`, output runtime; không sửa theo trí nhớ.
- [ ] **Đã cập nhật `docs/progress_status.md`** — Module Log (dòng mới), Decisions Log, Risks & Mitigations, Active Module → Status ✅ Commit, dòng "Cập nhật lần cuối".
- [ ] **Đã cập nhật `docs/AI_Usage.md`** — ghi thêm artifact mới (file, prompt, sửa đổi).
- [ ] `README.md` cập nhật nếu có thay đổi về cách chạy (kèm link `report.html` mới nhất ở mục Results).
- [ ] Module eval/visualize đã sinh `results/runs/<ts>/report.html` offline mở được bằng double-click, số khớp `metrics.json`.
- [ ] **Đã đối chiếu chéo** — grep các con số cũ (số lớp, số mẫu, shape) trong `docs/` và `README.md`, sửa hết chỗ lệch.
- [ ] **Đã verify** — mở lại từng file docs đã sửa, không còn câu chữ mâu thuẫn với code.
- [ ] Comment trong code cho phần phức tạp.

## Committed
- [ ] Commit theo Conventional Commit (`feat:`, `fix:`, `docs:`, `refactor:`, `test:`, `chore:`).
- [ ] Một module = một commit (hoặc nhóm commit nhỏ có liên quan).
- [ ] Message commit có phạm vi rõ ràng (`feat(data): thêm pipeline tiền xử lý`).
- [ ] Đã push lên remote (nếu làm việc nhóm).