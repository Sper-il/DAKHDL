# Prompt — Plan

Dùng khi bắt đầu một module mới. Mục tiêu: AI lập kế hoạch chi tiết, **chưa code**.

## Template

```
Bạn là AI Engineer. Hãy lập kế hoạch cho module dưới đây. KHÔNG viết code, chỉ phân tích.

# Ngữ cảnh project
- Tên project: <tên>
- Bài toán: <mô tả ngắn 1-2 câu>
- Tech stack: <vd: PyTorch 2.x, Albumentations, ONNX>
- Module liên quan: <danh sách module đã có>

# Module cần plan
- Tên module: <tên>
- Mục tiêu: <mô tả chức năng>
- Input: <mô tả input, kèm shape / định dạng>
- Output: <mô tả output, kèm shape / định dạng>

# Ràng buộc
- Phụ thuộc: <module / thư viện phải dùng>
- Không được dùng: <thư viện / pattern cấm>
- Performance budget: <latency, memory, accuracy tối thiểu>

# Yêu cầu đầu ra
1. Mục tiêu của module (1-2 câu).
2. Các bước con cần làm (liệt kê theo thứ tự).
3. Cấu trúc file / class / function đề xuất.
4. Interfaces / APIs của module (input/output kèm shape / định dạng, khớp với Architecture Snapshot).
5. Thư viện cần dùng + lý do.
6. Edge case cần xử lý.
7. Tiêu chí thành công (metric + threshold).
8. Milestones cho module (chia sub-task theo thứ tự + ước lượng).
9. Rủi ro kỹ thuật và cách giảm thiểu.
```

## Ví dụ đã điền

```
Bạn là AI Engineer. Hãy lập kế hoạch cho module dưới đây. KHÔNG viết code, chỉ phân tích.

# Ngữ cảnh project
- Tên project: Face Segmentation Lab
- Bài toán: Phân đoạn vùng mặt trong ảnh chân dung.
- Tech stack: PyTorch 2.x, Albumentations, segmentation_models_pytorch.
- Module liên quan: data loader (đã có), model (chưa có), trainer (chưa có).

# Module cần plan
- Tên module: data loader
- Mục tiêu: Load ảnh và mask, augment, trả về tensor sẵn sàng cho trainer.
- Input: đường dẫn ảnh PNG 512x512 RGB, mask nhị phân 512x512.
- Output: dict { 'image': tensor[3,H,W], 'mask': tensor[1,H,W] }.

# Ràng buộc
- Phụ thuộc: Albumentations, torchvision.
- Không được dùng: OpenCV cho augment (dùng Albumentations thôi).
- Performance budget: 100 mẫu/giây trên CPU.

# Yêu cầu đầu ra
1. Mục tiêu của module.
2. Các bước con.
3. Cấu trúc class / function.
4. Thư viện + lý do.
5. Edge case (ảnh corrupt, mask rỗng, kích thước lệch).
7. Tiêu chí thành công (ví dụ: > 100 mẫu/giây, mask khớp ảnh 100%).
```