# AI Usage

> Hồ sơ trung thực về cách các công cụ AI được sử dụng trong project này.

## 1. Tools Used
| Vai trò | Tool / Model | Phiên bản | Khi nào |
|---------|--------------|-----------|---------|
| Code generation | Cursor + Agent | 2026-11 | Scaffold thư mục, viết progress_status, README, docs, configs |
| Documentation | Cursor + Agent | 2026-11 | Soạn Define 6 mục và Design 7 mục theo skill |
| Review | Cursor + Agent | 2026-11 | Review skill define/design trước khi áp vào đề tài |
| Cross-check | Cursor + Agent | 2026-11-15 | Đối chiếu code ↔ docs sau khi mở rộng nhánh mặt lên 10 lớp |

## 2. Plain Purposes
- **Code generation** — scaffold khung project từ skill, viết file mẫu.
- **Documentation** — nháp Define, Design, README, architecture.
- **Debugging** — sửa lỗi IndentationError trong `face_cnn.py` do chỉnh sửa cùng lúc thêm test forward.
- **Research** — đối chiếu FER2013 + RAVDESS + late fusion với yêu cầu đề tài.
- **Review** — kiểm tra skill đủ Define/Design trước khi code.
- **Cross-check** — đọc trực tiếp `data/splits/*.csv` để lấy số liệu thật, rồi sửa docs theo số liệu đó (không sửa theo trí nhớ).

## 3. Student Modifications
- Tự tìm nguồn ảnh cho 3 lớp mới của nhánh mặt (AffectNet, Student Engagement) và viết `src/data/add_face_extra.py` để gộp vào splits.
- Chọn dùng ảnh `happy` từ AffectNet làm proxy cho `excited` — đã ghi rõ giới hạn này trong docs và Open Questions.
- Bổ sung rule vào skill `ai-fusion-workflow`: module chỉ xong khi docs đã cập nhật theo code.
- **Quyết định xây lại EVAL/VISUALIZE/DEMO** (2026-11-15) để khớp thang 10 lớp — đây là quyết định của sinh viên, AI thực hiện việc xác định đặc tả thiết kế và cập nhật docs cho khớp.

## 4. Important AI-Generated Artifacts
| File | Nội dung | Prompt đã dùng (tóm tắt) |
|------|----------|--------------------------|
| `docs/progress_status.md` | Define 6 mục + Design 7 mục cho emotion-fusion | Scaffold từ lựa chọn thang cảm xúc, FER+RAVDESS, late fusion |
| `README.md` | README 10 mục cho đề tài | Viết từ progress_status đã duyệt |
| `docs/architecture.md` | Sơ đồ late fusion 2 nhánh | Vẽ từ Design đã chốt |
| `configs/*.yaml` | Config data/model/train/eval | Sinh từ shape 48x48 và mel 128x128 |
| `src/models/fusion.py` | `_align_probs` căn theo `min` + docstring 10 lớp | Sửa để cả 2 nhánh cùng thang 10 lớp, giữ fallback ckpt 7 lớp |
| `src/demo/__main__.py`, `src/evaluation/__main__.py` | Suy `num_classes` từ shape checkpoint | Tránh hardcode số lớp, tương thích checkpoint cũ |
| `tests/unit/test_model_shapes.py` | 5 test: fusion 10 lớp, fallback 7 lớp, forward 2 backbone | Bổ sung test forward thật cho `FaceCNN`/`AudioCNN` (trước đó chỉ test fusion) |
| 10 file docs khác | Đồng bộ theo số liệu thật từ `data/splits/*.csv` | Yêu cầu AI quét thay đổi code → map sang docs bắt buộc cập nhật |
| `docs/{evaluation,visualization,demo}.md` | Đánh dấu CHƯA LÀM + giữ đặc tả thiết kế | Yêu cầu AI cập nhật docs sau khi xoá code, không để docs mô tả code không tồn tại |
| `docs/plan_remaining.md` | Plan 4 bước (1 Train+tuning · 2 Eval · 3 Visualize · 4 Demo); mỗi bước con ghi làm file gì · chức năng · làm thêm sau; kèm bảng files cần viết, phân công 4 người, timeline 6 ngày, DoD | Yêu cầu AI viết plan các bước chưa làm để giao cho thành viên, theo format từng bước cụ thể + rút gọn lại thành 4 bước dễ hiểu |
| `scripts/check_pending_modules.py` + `tests/unit/test_pending_modules.py` | Cơ chế tự động bảo đảm file của module chưa triển khai luôn trống/placeholder; chặn file rỗng tưởng đã làm xong, chặn import tới module chưa làm | Yêu cầu AI đảm bảo các code của các bước chưa làm luôn trống |
| `docs/{evaluation,visualization,demo}.md` | Đặc tả thiết kế cho 3 module chưa triển khai | Yêu cầu AI viết đặc tả thiết kế chi tiết để thành viên triển khai theo, không để docs mô tả code không tồn tại |

## 5. Prompt Purpose (cho mỗi artifact)
- **`docs/progress_status.md`** — yêu cầu AI điền đủ Define/Design từ 4 lựa chọn đã chốt, không bịa số liệu, mục chưa biết ghi TBD.
- **`README.md`** — yêu cầu AI viết 10 mục, đồng bộ metric với `docs/progress_status.md`.
- **`docs/`** — yêu cầu AI tách architecture, data-flow và api thành 3 file ngắn gọn.
- **`configs/`** — yêu cầu AI ghim shape, seed và đường dẫn đúng folder chuẩn.
- **Đợt đồng bộ 2026-11-15** — yêu cầu AI đọc số liệu thật từ `data/splits/*.csv` và `configs/`, đối chiếu từng câu trong docs, sửa các chỗ còn nói "7 lớp" thành "10 lớp".

## 6. Student Modifications
- Tự tìm nguồn ảnh cho 3 lớp mới của nhánh mặt (AffectNet, Student Engagement) và viết `src/data/add_face_extra.py` để gộp vào splits.
- Chọn dùng ảnh `happy` từ AffectNet làm proxy cho `excited` — đã ghi rõ giới hạn này trong docs và Open Questions.
- Bổ sung rule vào skill `ai-fusion-workflow`: module chỉ xong khi docs đã cập nhật theo code.

## 7. Validation Method
- [ ] Chạy code end-to-end trên môi trường sạch.
- [ ] So sánh output với baseline đã biết.
- [x] Viết unit/integration test mà không có thay đổi thì sẽ fail (7/7 test trong `tests/unit`, gồm forward thật 2 backbone).
- [ ] Reproduce kết quả từ fresh clone + ghi lại command.
- [x] Sanity-check metric theo Definition of Done.
- [x] Đối chiếu docs với dữ liệu thật trong `data/splits/*.csv` và `configs/*.yaml` (đợt 2026-11-15).
