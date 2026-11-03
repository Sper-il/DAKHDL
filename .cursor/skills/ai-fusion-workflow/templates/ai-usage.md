# AI_Usage.md — 7 mục chuẩn

Dùng làm khung cho `AI_Usage.md`. Ghi lại **trung thực** cách AI đã được dùng trong project.

```markdown
# AI Usage

> Hồ sơ trung thực về cách các công cụ AI được sử dụng trong project này.

## 1. Tools Used
| Vai trò | Tool / Model | Phiên bản | Khi nào |
|---------|--------------|-----------|---------|
| Code generation | Cursor + Claude | <ver> | Scaffold module, viết boilerplate |
| Documentation | <tool> | <ver> | Tóm tắt quyết định thiết kế |
| Debugging | <tool> | <ver> | Phân tích traceback, đề xuất root cause |
| Research | <tool> | <ver> | So sánh thư viện, tóm tắt paper |
| Review | <tool> | <ver> | Review code AI generate |

## 2. Plain Purposes
- **Code generation** — scaffold module từ thiết kế đã duyệt, viết hàm tiện ích.
- **Documentation** — nháp phần README, tóm tắt quyết định thiết kế.
- **Debugging** — đề xuất root cause, gợi ý log cần thêm.
- **Research** — so sánh thư viện, tóm tắt paper, lên kế hoạch experiment.
- **Review** — review code AI đã viết, phát hiện edge case.

## 3. Student Contributions (con người làm gì)
- Định nghĩa bài toán và phạm vi.
- Quyết định kiến trúc cuối cùng (kèm lý do).
- Toàn bộ phương pháp đánh giá và chọn metric.
- Toàn bộ review cuối cùng cho code AI generate.
- Toàn bộ commit (AI có thể nháp, người quyết định).
- Toàn bộ kết luận thí nghiệm.

## 4. Important AI-Generated Artifacts
| File | Nội dung | Prompt đã dùng (tóm tắt) |
|------|----------|--------------------------|
| <path> | <mô tả> | <link hoặc trích prompt> |
| <path> | <mô tả> | <link hoặc trích prompt> |
| <path> | <mô tả> | <link hoặc trích prompt> |

## 5. Prompt Purpose (cho mỗi artifact)
- **<file>** — prompt yêu cầu AI scaffold module data loader theo plan đã duyệt, dùng Albumentations thay vì torchvision.transforms.
- **<file>** — prompt yêu cầu AI giải thích tại sao training loss = NaN ở epoch 50, kèm full traceback.
- **<file>** — prompt yêu cầu AI review code module evaluation, tập trung vào edge case.

## 6. Student Modifications
- **<file>** — sửa logic augment để tránh flip mask ảo, vì mask nhị phân flip không tương đương flip ảnh trong một số trường hợp.
- **<file>** — thêm gradient clipping max_norm=1.0 sau khi AI đề xuất, vì training loss bị NaN.
- **<file>** — đổi optimizer từ Adam sang AdamW, vì paper gốc dùng AdamW và cho kết quả tốt hơn.

## 7. Validation Method
- [ ] Chạy code end-to-end trên môi trường sạch.
- [ ] So sánh output với baseline / đáp án đã biết.
- [ ] Viết unit/integration test mà không có thay đổi thì sẽ fail.
- [ ] Reproduce kết quả từ fresh clone + ghi lại command.
- [ ] Sanity-check metric theo Definition of Done.
```