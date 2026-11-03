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
- [ ] `README.md` cập nhật nếu có thay đổi về cách chạy (kèm link `report.html` mới nhất ở mục Results).
- [ ] `docs/AI_Usage.md` ghi thêm artifact mới (file, prompt, sửa đổi).
- [ ] `docs/progress_status.md` cập nhật module log, decisions log, risks.
- [ ] Module eval/visualize đã sinh `results/runs/<ts>/report.html` offline mở được bằng double-click, số khớp `metrics.json`.
- [ ] Comment trong code cho phần phức tạp.

## Committed
- [ ] Commit theo Conventional Commit (`feat:`, `fix:`, `docs:`, `refactor:`, `test:`, `chore:`).
- [ ] Một module = một commit (hoặc nhóm commit nhỏ có liên quan).
- [ ] Message commit có phạm vi rõ ràng (`feat(data): thêm pipeline tiền xử lý`).
- [ ] Đã push lên remote (nếu làm việc nhóm).