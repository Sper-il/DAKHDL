# Prompt — Implement

Dùng khi đã có plan và muốn AI viết code cho module. Sau khi nhận code, **luôn dùng prompt Review**.

## Template

```
Bạn là AI Engineer. Hãy viết code cho module dưới đây theo plan đã thống nhất.

# Plan đã thống nhất
<dán output từ prompt Plan, hoặc tóm tắt ngắn gọn>

# Yêu cầu kỹ thuật
- Ngôn ngữ: Python 3.10+
- Framework: <PyTorch / TensorFlow / sklearn>
- Style: PEP 8, type hint đầy đủ
- Logging: dùng module logging, không print rả
- Error handling: raise exception rõ ràng, có message có nghĩa

# Yêu cầu đầu ra
1. Code hoàn chỉnh, có thể copy-paste chạy được.
2. Docstring cho mỗi class / function public.
4. Ví dụ sử dụng ở cuối file (trong block `if __name__ == "__main__":`).
5. Giải thích ngắn (3-5 dòng) cho mỗi quyết định thiết kế quan trọng.

# KHÔNG ĐƯỢC
- KHÔNG thêm tính năng ngoài plan.
- KHÔNG dùng thư viện chưa có trong requirements.txt (trừ khi được duyệt).
- KHÔNG commit secrets, hardcoded path, absolute path.
- KHÔNG viết comment thừa ("# khởi tạo biến x" cho `x = 1`).
```

## Ví dụ đã điền

```
Bạn là AI Engineer. Hãy viết code cho module data loader theo plan.

# Plan đã thống nhất
- Class FaceSegDataset(torch.utils.data.Dataset).
- Đọc ảnh PNG và mask PNG từ thư mục data/processed/.
- Augment: horizontal flip, brightness/contrast, affine.
- Normalize về mean/std ImageNet.
- Output: dict { 'image': tensor[3,512,512], 'mask': tensor[1,512,512] }.

# Yêu cầu kỹ thuật
- Dùng Albumentations 1.3+, không dùng torchvision.transforms cho augment.
- DataLoader: num_workers=4, pin_memory=True.

# Yêu cầu đầu ra
1. File src/data/dataset.py.
2. Class + function đầy đủ.
3. Docstring test.
6. Ví dụ load 1 batch trong __main__.

# KHÔNG ĐƯỢC
- KHÔNG dùng OpenCV.
- KHÔNG hardcode path.
```