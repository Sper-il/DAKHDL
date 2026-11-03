# AI Usage

> Hồ sơ trung thực về cách các công cụ AI được sử dụng trong project này.

## 1. Tools Used
| Vai trò | Tool / Model | Phiên bản | Khi nào |
|---------|--------------|-----------|---------|
| Code generation | Cursor + Agent | 2026-11 | Scaffold thư mục, viết progress_status, README, docs, configs |
| Documentation | Cursor + Agent | 2026-11 | Soạn Define 6 mục và Design 7 mục theo skill |
| Review | Cursor + Agent | 2026-11 | Review skill define/design trước khi áp vào đề tài |

## 2. Plain Purposes
- **Code generation** — scaffold khung project từ skill, viết file mẫu.
- **Documentation** — nháp Define, Design, README, architecture.
- **Debugging** — chưa dùng (sẽ dùng khi train fail).
- **Research** — đối chiếu FER2013 + RAVDESS + late fusion với yêu cầu đề tài.
- **Review** — kiểm tra skill đủ Define/Design trước khi code.

## 3. Student Contributions (con người làm gì)
- Chốt đề tài và 4 lựa chọn: 7 cảm xúc, FER2013 + RAVDESS, late fusion, local GPU + HTML.
- Quyết định kiến trúc cuối cùng và ngưỡng metric.
- Toàn bộ tải dữ liệu, chạy train, xác minh số liệu.
- Toàn bộ review cuối cùng và mọi commit.
- Toàn bộ kết luận thí nghiệm trong báo cáo.

## 4. Important AI-Generated Artifacts
| File | Nội dung | Prompt đã dùng (tóm tắt) |
|------|----------|--------------------------|
| `docs/progress_status.md` | Define 6 mục + Design 7 mục cho emotion-fusion | Scaffold từ lựa chọn 7 cảm xúc, FER+RAVDESS, late fusion |
| `README.md` | README 10 mục cho đề tài | Viết từ progress_status đã duyệt |
| `docs/architecture.md` | Sơ đồ late fusion 2 nhánh | Vẽ từ Design đã chốt |
| `configs/*.yaml` | Config data/model/train/eval | Sinh từ shape 48x48 và mel 128x128 |

## 5. Prompt Purpose (cho mỗi artifact)
- **`docs/progress_status.md`** — yêu cầu AI điền đủ Define/Design từ 4 lựa chọn đã chốt, không bịa số liệu, mục chưa biết ghi TBD.
- **`README.md`** — yêu cầu AI viết 10 mục, đồng bộ metric với `docs/progress_status.md`.
- **`docs/`** — yêu cầu AI tách architecture, data-flow và api thành 3 file ngắn gọn.
- **`configs/`** — yêu cầu AI ghim shape, seed và đường dẫn đúng folder chuẩn.

## 6. Student Modifications
- (Sinh viên ghi sau khi review) — ví dụ: đổi trọng số fusion, đổi backbone, sửa threshold.
- (Sinh viên ghi sau khi chạy) — ví dụ: sửa batch size cho vừa VRAM lab.

## 7. Validation Method
- [ ] Chạy code end-to-end trên môi trường sạch.
- [ ] So sánh output với baseline đã biết.
- [ ] Viết unit/integration test mà không có thay đổi thì sẽ fail.
- [ ] Reproduce kết quả từ fresh clone + ghi lại command.
- [ ] Sanity-check metric theo Definition of Done.
