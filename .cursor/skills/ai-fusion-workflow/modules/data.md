# Data — Dữ liệu và data pipeline

Chuẩn làm việc với dữ liệu: từ raw đến splits sẵn sàng cho training, tránh leakage và mất vết.

## 1. Mục tiêu
- Mọi mẫu dữ liệu đều truy vết được: lấy từ đâu, xử lý gì, thuộc split nào.
- `src/data` chạy được độc lập: load một batch mẫu để kiểm tra shape trước khi đụng tới model.

## 2. Nội dung bắt buộc
- **Nguồn dữ liệu**: lấy từ đâu, giấy phép, ngày thu thập, ai chịu trách nhiệm.
- **Mô tả thống kê**: số mẫu, số lớp, phân bố lớp, kích thước ảnh / độ dài văn bản, tỉ lệ thiếu / corrupt.
- **Quy ước thư mục**: `data/raw/` bất biến, `data/processed/` đã làm sạch, `data/splits/` chứa train/val/test (tất cả gitignore).
- **Pre-processing**: resize, normalize (mean/std nào), chuẩn hóa văn bản, xử lý missing value.
- **Augmentation**: danh sách phép augment + tham số, chỉ augment trên train, không augment val/test.
- **Split**: tỉ lệ (ví dụ 70/15/15), split theo gì (random / theo thời gian / theo bệnh nhân để tránh leakage), seed cố định, file split lưu ở đâu.
- **Dataset / DataLoader**: class nào, output shape nào, `num_workers`, `pin_memory`, seed.

## 3. Template file
- `src/data/dataset.py` — class Dataset trả về `dict { 'input': ..., 'target': ... }`.
- `src/data/transforms.py` — hàm `build_train_transforms()` và `build_val_transforms()`.
- `src/data/split.py` — hàm `make_splits()` sinh file trong `data/splits/`.
- `src/data/__main__.py` — chạy `python -m src.data --config configs/data_config.yaml` để in shape 1 batch.
- `configs/data_config.yaml` — single source of truth cho đường dẫn, image size, batch size, mean/std.

## 4. Checklist review
- [ ] `data/raw/` có bao giờ bị sửa trực tiếp không? (phải là không)
- [ ] Val/test có lọt augment của train không?
- [ ] Có leakage không? (mẫu train xuất hiện ở val, cùng bệnh nhân / cùng video bị chia cắt)
- [ ] Split có cố định bằng seed + file không, hay mỗi lần chạy ra một kiểu?
- [ ] Chạy 1 batch có đúng shape đã khai trong Design (`Interfaces / APIs`) không?
- [ ] Ảnh corrupt, mask rỗng, label thiếu đã có cách xử lý / bỏ qua có log chưa?
- [ ] Dataset lớn có gitignore đúng, có ghi hướng dẫn tải trong `docs/data.md` chưa?

## 5. Rủi ro thường gặp
- Augment làm sai mask (flip / crop lệch giữa ảnh và mask) → dùng thư viện xử lý đồng thời ảnh-mask (ví dụ Albumentations) và visualize vài mẫu sau augment.
- Normalize sai mean/std → metric thấp khó hiểu → chốt mean/std trong config, ghi rõ nguồn (ImageNet hay tính trên tập train).
- Class imbalance không xử lý → model đoán toàn lớp đa số → dùng weighted sampler / class weight / focal loss, báo cáo phân bố lớp.
- Không version dataset → không reproduce được → đặt tên `{purpose}_{date}` và log checksum / số mẫu vào `results/runs/<ts>/config.yaml`.

## 6. Prompt mẫu
```
Bạn là AI Engineer. Hãy lập kế hoạch rồi viết code cho module data:
- Đọc configs/data_config.yaml (image size, batch size, split ratio, seed).
- Viết src/data/dataset.py, transforms.py, split.py theo plan đã duyệt.
- Output mỗi mẫu là dict với shape đã chốt trong Design.
- Không augment val/test. Xử lý ảnh corrupt bằng cách log và bỏ qua.
- Cuối file có block __main__ in shape 1 batch để verify nhanh.
```
