# Prompt — Review

Dùng sau khi AI generate code, trước khi commit. Mục tiêu: phát hiện lỗi, chỉ ra chỗ cần sửa, đề xuất cải thiện.

## Template

```
Bạn là code reviewer. Hãy review code Python dưới đây cho module <tên module>.

# Code cần review
```python
<paste code vào đây>
```

# Ngữ cảnh
- Module làm gì: <mô tả>
- Tech stack: <PyTorch / sklearn / ...>
- Convention: <PEP 8, type hint, docstring>

# Tiêu chí review (đánh dấu ✅ / ❌ / ⚠️ cho mỗi mục)
1. **Đúng yêu cầu** — code có giải quyết đúng bài toán không?
2. **Bug tiềm ẩn** — có chỗ nào có thể crash, gây memory leak, hoặc sai logic?
3. **Edge case** — input rỗng, None, ngoài phạm vi, kiểu lạ đã xử lý chưa?
5. **Performance** — có chỗ nào chậm bất hợp lý không? (vd: vòng lặp có thể vector hoá)
6. **Đọc được** — đặt tên biến / function có rõ không? Có comment đúng chỗ cần thiết không?
7. **Convention** — có tuân theo PEP 8, type hint, docstring không?
8. **Bảo mật** — có hardcode secret / path không? Có eval / exec không?
9. **Test** — có chỗ nào khó test không? Nên tách function nhỏ hơn?

# Yêu cầu đầu ra
1. Tóm tắt 1 đoạn (điểm mạnh, điểm yếu).
2. Danh sách vấn đề, sắp xếp theo mức độ nghiêm trọng (Critical → Major → Minor → Nitpick).
3. Với mỗi vấn đề: mô tả + đoạn code gốc + đề xuất sửa.
4. Gợi ý cải thiện (không bắt buộc).
```

## Ví dụ đã điền

```
Bạn là code reviewer. Hãy review code cho module data loader.

# Code
```python
import cv2
import numpy as np

class FaceDataset:
    def __init__(self, path):
        self.path = path
        self.images = os.listdir(path + "/images")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img = cv2.imread(self.path + "/images/" + self.images[idx])
        mask = cv2.imread(self.path + "/masks/" + self.images[idx], 0)
        return img, mask
```

# Ngữ cảnh
- Module: data loader cho face segmentation.
- Tech stack: PyTorch.
- Convention: PEP 8, type hint, docstring.

# Tiêu chí review
<như trên>
```