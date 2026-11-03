# Visualization — Trực quan hóa bằng HTML

Chuẩn trực quan hóa dữ liệu, quá trình train và kết quả đánh giá bằng một file HTML duy nhất, mở là xem được, không cần server.

## 1. Mục tiêu
- Mọi lần chạy quan trọng đều có một `report.html` mở được bằng trình duyệt, không cần cài thêm gì.
- Người không biết code vẫn hiểu được: dữ liệu thế nào, model học ra sao, sai ở đâu.
- Mọi ảnh và số liệu trong HTML đều sinh từ `results/runs/<ts>/`, không chế số thủ công.

## 2. Nguyên tắc
- **Một file duy nhất**: `results/runs/<ts>/report.html`, nhúng CSS và ảnh base64 vào trong, không phụ thuộc mạng.
- **Offline 100%**: không dùng CDN, không gọi API ngoài, mở bằng `double-click` là chạy.
- **Nhẹ**: ảnh resize tối đa cạnh 800 px, JPEG chất lượng 80 cho ảnh mẫu, PNG chỉ cho đồ thị.
- **Tái tạo được**: script sinh HTML nằm ở `src/evaluation/visualize_html.py`, chạy lại từ `metrics.json` + `config.yaml` là ra đúng file cũ.
- **Trung thực**: hiển thị cả case tốt và case tệ, metric đạt và chưa đạt, không chỉ khoe case đẹp.

## 3. Nội dung bắt buộc của report.html
1. **Header**: tên project, run id (`20261011-1708`), commit hash, thời gian chạy, link `config.yaml`.
2. **Tóm tắt Define**: problem một câu, input/output shape, metric target.
3. **Phân bố dữ liệu**: số mẫu train/val/test, biểu đồ cột phân bố lớp, 6–9 ảnh mẫu train (kèm mask/box nếu có).
4. **Quá trình train**: đồ thị loss train/val, metric val theo epoch, learning rate theo epoch.
5. **Kết quả model level**: bảng metric → target → đạt được → pass/fail, confusion matrix (classification) hoặc 6 ảnh so sánh dự đoán và ground truth (detection/segmentation).
6. **Kết quả pipeline**: latency end-to-end, throughput, VRAM, kích thước model.
7. **Error analysis**: 5 case tệ nhất kèm nguyên nhân đoán được và đề xuất sửa.
8. **Limitations + Next steps**: 3 gạch đầu dòng mỗi mục.

## 4. Template file
- `src/evaluation/visualize_html.py` — hàm `build_report(run_dir)` đọc `metrics.json` + ảnh trong `plots/` rồi render ra `report.html`.
- `templates/report.html` — khung HTML mẫu của skill, chỉ dùng CSS nội tuyến, không JavaScript nặng.
- `results/runs/<ts>/report.html` — output cuối, mở trực tiếp để xem hoặc nộp bài.
- `results/demo/<ts>/index.html` — bản rút gọn cho demo, chỉ giữ header, 3 case mẫu và bảng metric chính.

## 5. Cách sinh nhanh không cần thư viện nặng
- Vẽ đồ thị bằng `matplotlib`, lưu PNG rồi chuyển base64 nhúng vào HTML.
- Không bắt buộc `jinja2`: dùng `string.Template` hoặc f-string với file `templates/report.html` để thay placeholder.
- Nếu cần biểu đồ tương tác, dùng duy nhất một file `plotly.min.js` chép local, không dùng CDN.
- Tên ảnh trong `plots/` theo quy ước: `loss_curve.png`, `confusion_matrix.png`, `samples_good_*.jpg`, `samples_bad_*.jpg`.

## 6. Khung report.html tối thiểu
```html
<!DOCTYPE html>
<html lang="vi">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title><!--PROJECT--> — <!--RUN_ID--></title>
<style>
  body { font-family: Arial, sans-serif; max-width: 960px; margin: 24px auto; padding: 0 16px; color: #222; }
  table { border-collapse: collapse; width: 100%; }
  th, td { border: 1px solid #ccc; padding: 8px; text-align: left; }
  .pass { color: #0a7b2e; font-weight: bold; }
  .fail { color: #b00020; font-weight: bold; }
  img { max-width: 100%; height: auto; }
  .grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; }
</style>
</head>
<body>
  <h1><!--PROJECT--> — <!--RUN_ID--></h1>
  <p>Commit: <!--COMMIT--> | Thời gian: <!--TIME--></p>
  <h2>1. Kết quả chính</h2>
  <!--METRICS_TABLE-->
  <h2>2. Đồ thị train</h2>
  <!--LOSS_CURVE_IMG-->
  <h2>3. Mẫu dự đoán</h2>
  <div class="grid"><!--SAMPLES--></div>
  <h2>4. Case tệ nhất</h2>
  <!--BAD_CASES-->
</body>
</html>
```

## 7. Checklist review
- [ ] `report.html` mở bằng double-click trên máy khác vẫn đủ ảnh và số, không vỡ link?
- [ ] Số trong HTML có khớp `metrics.json`, không gõ tay?
- [ ] Có cả đồ thị loss, bảng metric pass/fail và ảnh mẫu tốt/xấu?
- [ ] Ảnh đã resize, tổng dung lượng `report.html` dưới 10 MB?
- [ ] Có ghi run id, commit hash và đường dẫn checkpoint để truy vết?
- [ ] Bản demo rút gọn (`results/demo/<ts>/index.html`) có đồng bộ số với bản full?

## 8. Rủi ro thường gặp
- Nhúng ảnh gốc 4K khiến HTML nặng hàng trăm MB → resize cạnh dài về 800 px trước khi nhúng.
- Dùng CDN cho CSS/JS, sang máy chấm không mạng thì trắng trang → mọi CSS/JS phải nội tuyến hoặc file local.
- Gõ số tay vào HTML cho đẹp → sai với `metrics.json` → luôn render từ file JSON bằng script.
- Chỉ khoe case đẹp → mất tin khi giảng viên thử case khó → bắt buộc mục case tệ nhất và limitations.

## 9. Prompt mẫu
```
Bạn là AI Engineer. Hãy viết src/evaluation/visualize_html.py:
- Đọc results/runs/<ts>/metrics.json, config.yaml và ảnh trong plots/.
- Render templates/report.html thành results/runs/<ts>/report.html bằng string.Template, không thêm dependency mới.
- Nhúng ảnh dạng base64, resize cạnh dài 800 px.
- Bảng metric có cột target, đạt được, pass/fail.
- Kèm 5 ảnh tốt nhất và 5 ảnh tệ nhất, mỗi ảnh có caption confidence và latency.
```
