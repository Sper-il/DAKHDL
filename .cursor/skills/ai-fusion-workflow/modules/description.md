# Description — Mô tả project / module

File này chuẩn hóa cách viết mô tả trước khi code: project làm gì, cho ai, phạm vi đến đâu.

## 1. Mục tiêu
- Một người mới đọc 2 phút hiểu được project giải bài toán gì, output là gì, đo bằng metric nào.
- Chốt phạm vi (scope) để tránh phình tính năng khi code.

## 2. Nội dung bắt buộc
- **Tên + tagline**: tên đầy đủ, slug, một câu mô tả.
- **Problem**: bài toán cụ thể, vì sao quan trọng, ví dụ minh họa.
- **Target users**: ai dùng, bối cảnh nào, kỳ vọng gì ở output.
- **Input / Output**: kiểu dữ liệu, shape / định dạng, ví dụ concrete.
- **Core features**: bảng Must / Should / Could + lý do từng dòng.
- **Out of scope**: liệt kê rõ cái gì KHÔNG làm trong MVP.
- **Success criteria**: metric + threshold, ví dụ `Accuracy ≥ 0,90`, `Latency ≤ 50 ms`.

## 3. Template mô tả module
```markdown
## Module: <tên>
- Làm gì: <1-2 câu>
- Input: <định dạng + shape>
- Output: <định dạng + shape>
- Tiêu chí thành công: <metric + threshold>
- Ngoài phạm vi: <cái gì không làm>
```

## 4. Checklist review mô tả
- [ ] Đọc xong có hình dung được input → output concrete không?
- [ ] Có phân biệt Must / Should / Could không?
- [ ] Có mục out of scope không?
- [ ] Có số liệu (metric, threshold, shape) hay toàn chữ chung chung?
- [ ] Đã đồng bộ vào `docs/progress_status.md` mục Project Context chưa?

## 5. Rủi ro thường gặp
- Mô tả chung chung ("xử lý ảnh", "dự đoán tốt") → yêu cầu ví dụ input/output thật.
- Nhồi quá nhiều Could vào MVP → cắt, đẩy sang milestone sau.
- Thiếu threshold đo lường → mọi đánh giá sau đều cảm tính.

## 6. Prompt mẫu
```
Bạn là Product Owner. Dựa trên thông tin Define (project name, problem, users,
input/output, core features), hãy viết mô tả project 300-500 từ gồm:
tagline, problem, users, input/output ví dụ, bảng Must/Should/Could,
out of scope, success criteria (metric + threshold). Không viết code.
```
