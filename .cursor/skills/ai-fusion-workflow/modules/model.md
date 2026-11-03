# Model — Kiến trúc model

Chuẩn chọn và viết kiến trúc model: bắt đầu từ baseline đơn giản, version rõ ràng, config hóa toàn bộ.

## 1. Mục tiêu
- Mọi lựa chọn kiến trúc đều có lý do ghi trong `docs/progress_status.md` (Decisions Log).
- Model tái tạo được từ config + checkpoint, không phụ thuộc code cứng trong notebook.

## 2. Nội dung bắt buộc
- **Bài toán → họ model**: classification / detection / segmentation / tabular / NLP tương ứng với họ kiến trúc nào, vì sao chọn họ đó.
- **Baseline trước**: model đơn giản nhất chạy được là gì (ví dụ ResNet-18, UNet nhỏ, Logistic Regression), metric baseline bao nhiêu.
- **Backbone + head**: dùng backbone nào (pretrained ở đâu), head nào, input/output shape.
- **Loss function**: dùng loss gì, vì sao (cross-entropy, focal, dice, combo), class weight nếu mất cân bằng.
- **Số tham số / dung lượng**: ước lượng params, dung lượng file, có vừa ngân sách deploy không.
- **Config hóa**: mọi siêu tham số kiến trúc nằm trong `configs/model_config.yaml`, không hardcode trong code.
- **Naming**: `{domain}_{architecture}_{version}.{ext}`, ví dụ `face_unet_v1.pt`, kèm CHANGELOG khi đổi version.

## 3. Template file
- `src/models/architecture.py` — hàm `build_model(config)` trả về `nn.Module`, có docstring ghi input/output shape.
- `src/models/loss.py` — hàm `build_loss(config)` tách riêng để dễ thử nghiệm.
- `models/configs/` — lưu config kiến trúc đã dùng cho mỗi version.
- `models/trained/` — chỉ chứa model cuối (gitignore hoặc LFS), checkpoint giữa chừng nằm ở `models/checkpoints/`.

## 4. Checklist review
- [ ] Có baseline đơn giản chạy trước khi nâng cấp backbone lớn không?
- [ ] Pretrained weight lấy từ đâu, giấy phép có cho dùng không?
- [ ] Input/output shape của model có khớp `Interfaces / APIs` trong Design không?
- [ ] Loss có xử lý class imbalance / mask rỗng không?
- [ ] Mọi số (num_classes, in_channels, dropout) có nằm trong config không?
- [ ] Có ước lượng params / VRAM / inference time trước khi train dài không?
- [ ] Quyết định chọn / loại model có ghi vào Decisions Log không?

## 5. Rủi ro thường gặp
- Chọn model quá lớn so với dữ liệu nhỏ → overfit nhanh → bắt đầu nhỏ, tăng dần, luôn có weight decay + early stopping.
- Quên `model.train()` / `model.eval()` và `torch.no_grad()` khi eval → metric sai → checklist review phải soi.
- Mismatch số kênh / số lớp giữa data và model → crash hoặc học sai → verify bằng 1 batch thật trước khi train.
- Không seed / không version → không so sánh được experiment → cố định seed, tag version mỗi lần đổi kiến trúc.

## 6. Prompt mẫu
```
Bạn là AI Architect. Hãy đề xuất kiến trúc cho bài toán <mô tả> với ràng buộc
<params tối đa, latency tối đa, phần cứng>:
1. Baseline đề xuất + lý do.
2. 2 phương án nâng cấp + so sánh (accuracy, tốc độ, VRAM).
3. Loss đề xuất + cách xử lý mất cân bằng.
4. Cấu trúc file src/models/architecture.py và loss.py.
Không viết code chi tiết, chỉ phân tích để tôi duyệt.
```
