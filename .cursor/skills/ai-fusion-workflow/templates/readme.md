# README — 10 mục chuẩn

Dùng làm khung cho `README.md` của mọi project AI/ML.

```markdown
# <Tên Project>

> <Câu tagline 1 dòng, mô tả ngắn gọn project làm gì>

## 1. Project Overview
<2–3 đoạn: project là gì, dành cho ai, tại sao quan trọng ngay bây giờ.>
<Có thể kèm ảnh minh hoạ, GIF demo.>

## 2. Problem Statement
<Bài toán cụ thể, nên định lượng nếu có thể. Trích nguồn nỗi đau.>
<Ví dụ: "Phân loại ảnh X-quang phổi thành 3 lớp với độ chính xác > 95%, hỗ trợ bác sĩ tại các bệnh viện vùng sâu.">

## 3. Main Features
| Độ ưu tiên | Tính năng | Lý do |
|------------|-----------|-------|
| Must | <tính năng> | <lý do phải có> |
| Should | <tính năng> | <lý do nên có> |
| Could | <tính năng> | <lý do có thì tốt> |

## 4. System Architecture
<Sơ đồ tổng thể dạng text hoặc ảnh.>

```
[User] → [Ingest] → [Preprocess] → [Model] → [Postprocess] → [Output]
```

<Xem chi tiết trong docs/architecture.md.>

## 5. Folder Structure
<Sơ đồ cây thư mục chính, kèm mô tả ngắn cho mỗi thư mục.>

```
project/
├── data/        # dữ liệu (gitignore)
├── src/         # source code
├── models/      # model đã train
├── notebooks/   # EDA
├── tests/       # test
├── docs/        # tài liệu
├── configs/     # config
└── results/     # kết quả mỗi lần chạy
```

## 6. Installation / Run / Test
```bash
# 1. Clone
git clone <url> & cd <project>

# 2. Tạo virtual env
python -m venv .venv
source .venv/bin/activate   # Linux/Mac
.venv\Scripts\activate      # Windows

# 3. Cài dependency
pip install -r requirements.txt

# 4. Chuẩn bị data
# Đặt dữ liệu vào data/raw/ theo hướng dẫn trong docs/data.md

# 5. Train
python -m src.training --config configs/train_config.yaml

# 6. Evaluate
python -m src.evaluation --config configs/eval_config.yaml

# 7. Test
pytest -q
```

## 7. Example Input → Output
**Input**
```json
{
  "image_path": "data/processed/test/sample_001.png",
  "task": "segmentation"
}
```

**Output**
```json
{
  "mask_path": "results/runs/20261011-1708/predictions/sample_001.png",
  "confidence": 0.94,
  "inference_time_ms": 23
}
```

## 8. Results
Lần chạy mới nhất: `results/runs/<timestamp>/metrics.json` + báo cáo HTML `results/runs/<timestamp>/report.html` (mở bằng double-click, xem `modules/visualization.md`).

| Metric | Target | Đạt được | Run |
|--------|--------|----------|-----|
| Accuracy | ≥ 0.90 | 0.93 | 20261011-1708 |
| Latency | ≤ 50 ms | 23 ms | 20261011-1708 |
| mAP@0.5 | ≥ 0.85 | TBD | — |

## 9. Limitations
- <Hạn chế 1: vd chỉ chạy trên GPU NVIDIA, không support AMD.>
- <Hạn chế 2: vd accuracy giảm khi ảnh mờ.>
- <Hạn chế 3: vd chưa test trên domain khác.>

## 10. Future Improvements
- <Cải tiến 1: vd thêm data augmentation cho low-light.>
- <Cải tiến 2: vd export sang ONNX để chạy edge.>
- <Cải tiến 3: vd thêm API RESTful.>

---

Được duy trì với skill **ai-fusion-workflow**.
Xem `docs/AI_Usage.md` để biết cách AI được sử dụng trong project.
```