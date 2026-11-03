# Training — Vòng lặp huấn luyện

Chuẩn train reproducible: config hóa, log đầy đủ, checkpoint an toàn, dừng đúng lúc.

## 1. Mục tiêu
- Chạy lại từ `results/runs/<ts>/config.yaml` + seed là ra kết quả tương đương.
- Mọi lần train đều sinh một folder run mới, không bao giờ ghi đè kết quả cũ.

## 2. Nội dung bắt buộc
- **Config train**: optimizer, learning rate, scheduler, batch size, epoch, weight decay, gradient clipping, seed — tất cả trong `configs/train_config.yaml`.
- **Trainer**: `src/training/trainer.py` tách `train_one_epoch()` và `validate()`, dùng `logging` thay vì `print`.
- **Scheduler + early stopping**: loại scheduler, patience bao nhiêu, monitor metric nào, best checkpoint lưu theo metric nào.
- **Checkpoint**: lưu mỗi epoch hay chỉ best/last, đường dẫn `models/checkpoints/`, có lưu optimizer state để resume không.
- **Logging**: loss / metric mỗi epoch, learning rate hiện tại, thời gian mỗi epoch, GPU RAM — ghi ra `logs.txt` và TensorBoard / MLflow / W&B.
- **Resume**: có chạy tiếp từ checkpoint được không bằng flag `--resume`.
- **Tiền kiểm trước train dài**: overfit 1 batch nhỏ (sanity check), 1 epoch thử trên subset để ước lượng thời gian.

## 3. Template file
- `src/training/trainer.py` — class `Trainer` với `fit()`, `train_one_epoch()`, `validate()`, `save_checkpoint()`.
- `src/training/scheduler.py` — hàm `build_scheduler()` nếu scheduler phức tạp.
- `src/training/__main__.py` — CLI `python -m src.training --config configs/train_config.yaml`.
- `results/runs/YYYYMMDD-HHMM/` — mỗi lần chạy chứa `config.yaml`, `logs.txt`, `metrics.json`, `plots/loss_curve.png`.

## 4. Checklist review
- [ ] Overfit 1 batch nhỏ thành công chưa trước khi train full?
- [ ] Seed đã cố định (Python, NumPy, PyTorch) và ghi vào config chưa?
- [ ] Val dùng `model.eval()` + `torch.no_grad()` chưa?
- [ ] Best checkpoint chọn theo metric val, không phải loss train?
- [ ] Có gradient clipping khi loss có nguy cơ NaN không?
- [ ] Có early stopping để tránh train vô ích không?
- [ ] Folder run mới cho mỗi lần chạy, không sửa folder cũ?

## 5. Rủi ro thường gặp
- Loss NaN giữa chừng → kiểm tra learning rate quá cao, thiếu clipping, mixed precision overflow; xem `prompts/debug.md` để gỡ theo quy trình.
- Train/val gap lớn (overfit) → tăng augmentation, weight decay, dropout, hoặc giảm độ lớn model.
- Quên lưu optimizer khi cần resume → phải train lại từ đầu → luôn lưu `model + optimizer + scheduler + epoch + best_metric`.
- Không ước lượng thời gian → train kẹt GPU nhiều ngày → chạy thử 1 epoch trên subset rồi nhân lên trước khi commit train dài.

## 6. Prompt mẫu
```
Bạn là AI Engineer. Hãy viết src/training/trainer.py theo plan:
- Đọc configs/train_config.yaml (optimizer, lr, scheduler, epochs, seed).
- Tách train_one_epoch / validate, dùng logging, hỗ trợ --resume.
- Lưu best checkpoint theo <metric>, early stopping patience <N>.
- Mỗi epoch log loss, metric, lr, thời gian.
- Kèm sanity check overfit 1 batch trong docstring hướng dẫn.
```
