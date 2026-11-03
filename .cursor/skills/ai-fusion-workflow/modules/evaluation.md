# Evaluation — Đánh giá model và pipeline

Chuẩn đánh giá 2 lớp: model level (độ đúng) và full pipeline (tốc độ, tài nguyên, độ ổn định).

## 1. Mục tiêu
- Mọi con số báo cáo đều tái tạo được từ `results/runs/<ts>/metrics.json` + script eval.
- Không chỉ báo accuracy mà còn báo latency, tài nguyên và case thất bại.

## 2. Nội dung bắt buộc
- **Model level** (tùy bài toán):
  - Classification: accuracy, precision, recall, F1 + confusion matrix.
  - Detection: mAP@0.5, mAP@0.5:0.95 + ảnh visualize box.
  - Segmentation: IoU / Dice, Hausdorff + ảnh mask dự đoán vs ground truth.
  - Chung: inference time / mẫu, dung lượng model, VRAM.
- **Full pipeline**: latency end-to-end (preprocess + inference + postprocess), throughput (request/giây), CPU/GPU/RAM/disk, xử lý lỗi và phục hồi.
- **Ngưỡng đạt**: bảng metric → target → đạt được → run id, ví dụ `Accuracy ≥ 0,90 | 0,93 | 20261011-1708`.
- **Error analysis**: top case sai, gom nhóm nguyên nhân (ảnh mờ, lớp hiếm, label nhiễu), đề xuất cải thiện.
- **Artifact**: `metrics.json`, `plots/` (loss curve, confusion matrix), vài ảnh / file output mẫu.

## 3. Template file
- `src/evaluation/metrics.py` — hàm tính metric cho từng bài toán, có unit test.
- `src/evaluation/visualize.py` — hàm vẽ confusion matrix, overlay mask/box.
- `src/evaluation/__main__.py` — CLI `python -m src.evaluation --config configs/eval_config.yaml --checkpoint <path>`.
- `results/runs/<ts>/metrics.json` — schema cố định: `{ metric, value, target, passed, config, commit }`.

## 4. Checklist review
- [ ] Metric tính trên test set giữ riêng, không phải val đã tuning?
- [ ] Có báo kèm threshold / target, không chỉ số trần?
- [ ] Có confusion matrix hoặc visualize lỗi, không chỉ 1 con số?
- [ ] Latency đo trên đúng phần cứng mục tiêu (CPU-only vs GPU)?
- [ ] Có đo end-to-end, không chỉ thời gian forward model?
- [ ] Kết quả đã ghi vào `results/runs/<ts>/` và cập nhật `README` bảng Results chưa?
- [ ] Có phân tích case sai và đề xuất cải thiện concrete không?

## 5. Rủi ro thường gặp
- Tuning trên test → số đẹp nhưng ảo → khóa test, chỉ eval cuối cùng một lần, mọi thử nghiệm trên val.
- Chỉ đo trên GPU mạnh nhưng deploy CPU yếu → latency vỡ ngân sách → đo trên đúng môi trường deploy.
- Metric trung bình che giấu lớp hiếm kém → luôn báo per-class metric + confusion matrix.
- Không seed / không ghi config eval → không reproduce → mọi eval phải kèm config + commit hash + checkpoint path.

## 6. Prompt mẫu
```
Bạn là AI Engineer. Hãy viết module evaluation:
- Đọc checkpoint <path> và configs/eval_config.yaml.
- Tính <metric + target> trên test set, xuất metrics.json đúng schema.
- Vẽ confusion matrix + lưu 5 ảnh dự đoán tốt nhất / tệ nhất vào plots/.
- In bảng so sánh target vs đạt được và nhận xét pass/fail từng metric.
```
