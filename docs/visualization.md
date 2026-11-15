# Visualization — Trực quan hóa bằng report.html offline

> Chuẩn render báo cáo một file HTML duy nhất, mở bằng double-click, không cần mạng.

## Trạng thái: CHƯA LÀM

> Module này chưa được triển khai. Tài liệu giữ lại làm **đặc tả thiết kế** cho phần sắp xếp.
> Các file dưới đây đã tồn tại **rỗng (0 byte)** để thành viên mở ra viết.

| File (đã tồn tại, rỗng) | Nội dung |
|---|---|
| `src/evaluation/visualize.py` | `plot_confusion_matrix()`, `plot_loss_curve()` — matplotlib, xuất PNG |
| `src/evaluation/visualize_html.py` | `build_report()`, `metrics_table()`, `_img_b64()` — render 1 file HTML |

## 1. Nguyên tắc
- Một lần chạy chỉ sinh một `report.html` duy nhất trong `results/runs/<ts>/`.
- Offline 100%: không CDN, không font ngoài, ảnh nhúng base64, CSS inline.
- Số trong HTML phải khớp `metrics.json`, không sửa tay.
- Dung lượng mục tiêu dưới 10 MB để nộp bài và chấm offline.

## 2. Luồng render
```
metrics.json + plots/*.png + config.yaml
  → src/evaluation/visualize_html.py::build_report(run_dir)
  → results/runs/<ts>/report.html
```

Khung HTML: `.cursor/skills/ai-fusion-workflow/templates/report.html`.

Hàm chính:
- `_img_b64(path, max_side=800)`: resize cạnh dài về 800, nén JPEG quality 80, nhúng base64.
- `metrics_table(metrics)`: bảng nhánh, target, accuracy, kết luận ĐẠT/CHƯA ĐẠT.
- `build_report(run_dir, template_path)`: đọc JSON, thay placeholder `PROJECT, RUN_ID, COMMIT, TIME, CHECKPOINT, METRICS_TABLE, LOSS_CURVE_IMG, GOOD_SAMPLES, BAD_CASES, E2E_LATENCY_MS...`, ghi `report.html`.

## 3. Nội dung bắt buộc trong report.html
- Bảng metric pass/fail theo `configs/eval_config.yaml`.
- Đồ thị loss train/val (`*loss*.png` nếu có).
- Confusion matrix từng nhánh (`*confusion*.png`).
- Mẫu tốt (good samples) và 5 case tệ nhất (bad cases).
- Latency end-to-end, throughput, VRAM nếu đo được.
- Limitations và next steps cụ thể.

## 4. Lệnh chạy (sẽ khả dụng lại khi viết lại code)
```bash
# Eval đã tự gọi build_report, chỉ cần mở file
python -m src.evaluation \
  --face-ckpt models/checkpoints/face/best.pt \
  --face-test-csv data/splits/face_test.csv

# Mở báo cáo
start results/runs/<ts>/report.html
```

Kiểm tra: copy `report.html` sang máy khác không có mạng, double-click vẫn hiện đủ bảng và ảnh.

## 5. Xử lý sự cố
- Thiếu ảnh: placeholder hiện "Chưa có ảnh mẫu", cần chạy `visualize.py` để sinh plots trước.
- File quá nặng: giảm `max_side`, giảm quality JPEG, giới hạn số ảnh mẫu còn 6.
- Số lệch giữa HTML và JSON: luôn render lại từ `build_report`, không sửa HTML tay.

## 6. Điểm cần cải thiện khi viết lại

Run cũ `results/runs/20261109-1704/report.html` chỉ **2,7 KB** — thiếu gần hết nội dung bắt buộc ở mục 3 (không có đồ thị loss, không có mẫu tốt/xấu, không có case tệ nhất). Khi viết lại cần bảo đảm đủ 6 nhóm nội dung nêu trên.

> Lưu ý: `results/eda/report.html` (báo cáo EDA, ~5,1 MB) là file **khác**, do `notebooks/2026-11-03_eda_html.py` sinh ra và vẫn hoạt động bình thườên.
