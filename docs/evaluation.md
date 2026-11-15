# Evaluation — Đánh giá 2 lớp Emotion Fusion

> Tài liệu đánh giá model + pipeline. Đồng bộ với `configs/eval_config.yaml` và `src/evaluation/`.

## Trạng thái: CHƯA LÀM

> Module này chưa được triển khai. Tài liệu giữ lại làm **đặc tả thiết kế** cho phần sắp xếp.
> Các file dưới đây đã tồn tại **rỗng (0 byte)** để thành viên mở ra viết.

Các file cần viết:

| File (đã tồn tại, rỗng) | Nội dung |
|---|---|
| `src/evaluation/__main__.py` | CLI eval, sinh `metrics.json` + `report.html` |
| `src/evaluation/metrics.py` | `evaluate_branch()` + `evaluate_fusion()` |
| `src/evaluation/visualize.py` | `plot_confusion_matrix()` + `plot_loss_curve()` |
| `src/evaluation/visualize_html.py` | `build_report()` render HTML offline |

## 1. Hai lớp đánh giá
- **Model level**: accuracy, F1 macro, confusion matrix, inference time trên từng nhánh.
- **Full pipeline**: latency end-to-end (preprocess + inference + fusion), ngưỡng đạt/không đạt, phân tích lỗi.
- Mọi con số báo cáo đều tái tạo được từ `results/runs/<ts>/metrics.json` + checkpoint + test split.

## 2. Config đánh giá
File `configs/eval_config.yaml` (đã có, chưa có code đọc):

```yaml
metrics: [accuracy, f1_macro, confusion_matrix, inference_time_ms]
targets:
  accuracy_face: 0.65
  accuracy_audio: 0.70
  accuracy_fusion: 0.70
  e2e_latency_ms: 200
report_html: results/runs/<ts>/report.html
```

Ngưỡng pass cuối: fusion ≥ 0,70 và hơn nhánh đơn tốt nhất ≥ 3 điểm, F1 macro ≥ 0,60.

## 3. Hàm metric cần viết (`src/evaluation/metrics.py`)

- `evaluate_branch(model, loader, device)`: accuracy, F1 macro, confusion matrix (mặt 10x10, tiếng 10x10), `inference_time_ms` trung bình trên từng mẫu (đồng bộ CUDA khi đo).
- `evaluate_fusion(face_model, audio_model, loader_pairs, w_face, w_audio)`: yêu cầu cặp face/audio cùng nhãn trên 10 lớp chung, assert `yf == ya`, gộp qua `fuse_probs`, trả về accuracy, F1, confusion 10x10, latency, trọng số đã dùng.

## 4. Lệnh chạy (sẽ khả dụng lại khi viết lại code)
```bash
# Tối thiểu nhánh mặt
python -m src.evaluation \
  --face-ckpt models/checkpoints/face/best.pt \
  --face-test-csv data/splits/face_test.csv

# Cả 2 nhánh
python -m src.evaluation \
  --face-ckpt models/checkpoints/face/best.pt \
  --face-test-csv data/splits/face_test.csv \
  --audio-ckpt models/checkpoints/audio/best.pt \
  --audio-test-csv data/splits/audio_test.csv \
  --run-dir results/runs/<ts>
```

Output:
```
results/runs/<ts>/
├── metrics.json   # {time, branches: {face, audio}, e2e_latency_ms, error_analysis}
├── plots/
│   ├── confusion_face.png
│   └── confusion_audio.png
└── report.html    # sinh ở bước visualize
```

Hàm vẽ: `src/evaluation/visualize.py::plot_confusion_matrix`.

## 5. Đọc metrics.json
```json
{
  "time": "20261101-1900",
  "branches": {
    "face": {"accuracy": 0.66, "target": 0.65, "passed": true, "f1_macro": 0.61},
    "audio": {"accuracy": 0.71, "target": 0.70, "passed": true, "f1_macro": 0.68}
  },
  "e2e_latency_ms": 42.5
}
```

Bảng pass/fail render lại trong `report.html`, số phải khớp 100% với JSON.

## 6. Quy tắc chống gian lận metric
- Metric chỉ tính trên test set giữ riêng, mọi thử nghiệm trong lúc phát triển chỉ dùng val.
- Không tune trọng số fusion trên test, chỉ tune trên val (`tune_on_val: true`).
- Luôn báo per-class metric + confusion matrix, không chỉ báo accuracy trung bình.
- Đo latency trên đúng phần cứng mục tiêu (GPU lab, fallback CPU-only ghi rõ).

## 7. Output cũ còn trong repo (tham chiếu, KHÔNG phải kết quả hợp lệ)

`results/runs/20261109-1704/` là output của lần eval cũ, chạy trên model **7 lớp** trước khi mở rộng lên 10 lớp:
- Confusion matrix 7×7, không có nhánh fusion.
- Face accuracy 0,176 / audio accuracy 0,387 — đều fail.
- `report.html` chỉ 2,7 KB, thiếu đồ thị loss, mẫu tốt/xấu, case tệ nhất.

**Không được trích dẫn các số này trong báo cáo.** Cần eval lại sau khi viết lại code và train full.
