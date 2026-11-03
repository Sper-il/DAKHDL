# Prompt — Debug

Dùng khi code chạy sai, test fail, hoặc model cho kết quả không như kỳ vọng. Mục tiêu: tìm root cause nhanh, không "thử lung tung".

## Template

```
Bạn là AI Engineer giỏi debug. Hãy giúp tôi tìm nguyên nhân lỗi dưới đây.

# Lỗi gì?
<mô tả lỗi bằng lời: exception, output sai, model không học, training chậm...>

# Thông báo lỗi / output lạ
```
<paste full traceback hoặc output>
```

# Code liên quan
```python
<paste file hoặc đoạn code liên quan>
```

# Đã thử gì?
- <vd: in shape tensor, kiểm tra path, kiểm tra learning rate>
- <vd: giảm batch size, đổi optimizer>
- <các bước đã làm và kết quả>

# Kỳ vọng vs thực tế
- Kỳ vọng: <output / metric mong muốn>
- Thực tế: <output / metric đang có>

# Yêu cầu
1. **Phân tích root cause** (không đoán mò — đọc code + traceback thật kỹ).
2. **Đề xuất 2-3 cách sửa**, sắp xếp theo độ an toàn (an toàn nhất trước).
3. **Với mỗi cách**: mô tả thay đổi, rủi ro, cách verify.
4. **Cách verify nhanh** để biết đã fix đúng (vd: in shape, log metric, chạy 1 batch).

# KHÔNG ĐƯỢC
- KHÔNG đề xuất "thêm try/except để nuốt lỗi".
- KHÔNG đề xuất "tăng epoch lên xem sao".
- KHÔNG đoán nếu không có căn cứ từ code / log.
```

## Ví dụ đã điền

```
Bạn là AI Engineer. Giúp tôi debug lỗi training loss = NaN.

# Lỗi
Sau ~50 epoch, training loss đột ngột thành NaN, val loss cũng NaN.

# Thông báo lỗi
UserWarning: NaN encountered in loss.
Không có exception, chỉ là NaN.

# Code
```python
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss = F.binary_cross_entropy_with_logits(logits, masks)
loss.backward()
optimizer.step()
```

# Đã thử
- Giảm lr xuống 1e-4 — vẫn NaN ở epoch ~80.
- Thêm gradient clipping max_norm=1.0 — vẫn NaN.

# Kỳ vọng vs thực tế
- Kỳ vọng: loss giảm dần về ~0.1.
- Thực tế: loss giảm bình thường đến epoch 50, sau đó NaN.

# Yêu cầu
<như trên>
```