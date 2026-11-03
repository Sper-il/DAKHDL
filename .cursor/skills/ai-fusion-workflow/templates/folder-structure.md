# Cấu trúc thư mục chuẩn cho project AI/ML

```
project-root/
├── data/
│   ├── raw/                # dữ liệu gốc, không bao giờ sửa (gitignore)
│   ├── processed/          # dữ liệu đã xử lý sạch (gitignore)
│   └── splits/             # train/val/test split (gitignore)
│
├── src/
│   ├── data/               # data pipeline: load, augment, split
│   │   ├── dataset.py
│   │   ├── transforms.py
│   │   └── split.py
│   ├── models/             # kiến trúc model
│   │   ├── architecture.py
│   │   └── loss.py
│   ├── training/           # training loop
│   │   ├── trainer.py
│   │   └── scheduler.py
│   └── evaluation/         # đánh giá model + pipeline + visualize HTML
│       ├── metrics.py
│       ├── visualize.py
│       └── visualize_html.py   # render report.html offline từ metrics.json + plots/
│
├── models/                 # model đã train
│   ├── checkpoints/        # checkpoint trong quá trình train (gitignore)
│   ├── trained/            # model cuối cùng (gitignore hoặc LFS)
│   └── configs/            # config kiến trúc (yaml/json)
│
├── notebooks/              # EDA, thử nghiệm nhanh (đặt tên theo ngày)
│   └── 2026-10-11_eda.ipynb
│
├── tests/
│   ├── unit/               # test cho function / class
│   └── integration/        # test cho end-to-end flow
│
├── docs/
│   ├── architecture.md     # sơ đồ kiến trúc
│   ├── data-flow.md        # luồng dữ liệu
│   ├── api.md              # API nội bộ giữa các module
│   ├── description.md      # mô tả project (Define 6 mục)
│   ├── data.md             # quy ước dữ liệu
│   ├── progress_status.md  # single source of truth Define + Design
│   ├── AI_Usage.md         # hồ sơ sử dụng AI
│   ├── CHANGELOG.md        # lịch sử version
│   └── decisions.md        # log quyết định kỹ thuật (optional)
│
├── configs/                # config tổng thể (yaml/json)
│   ├── data_config.yaml
│   ├── model_config.yaml
│   └── train_config.yaml
│
├── results/
│   ├── runs/
│   │       └── 20261011-1708/  # theo timestamp
│           ├── metrics.json
│           ├── config.yaml
│           ├── logs.txt
│           ├── report.html         # báo cáo HTML offline duy nhất (xem modules/visualization.md)
│           └── plots/
│               ├── loss_curve.png
│               └── confusion_matrix.png
│   └── demo/
│       └── <timestamp>/          # output demo 1 lệnh (xem modules/demo.md)
│           ├── index.html        # bản rút gọn của report.html
│           └── metrics.json
│
├── .gitignore
├── README.md
├── docs/AI_Usage.md
├── docs/progress_status.md
├── requirements.txt
└── docs/CHANGELOG.md
```

## Nguyên tắc

- **`data/`** — gitignore hoàn toàn (file lớn, không track).
- **`models/checkpoints/`, `models/trained/`** — gitignore, dùng DVC hoặc LFS nếu cần share.
- **`notebooks/`** — đặt tên theo ngày `YYYY-MM-DD_mota.ipynb`.
- **`results/runs/<timestamp>/`** — mỗi lần chạy sinh ra một folder, không bao giờ sửa folder cũ.
- **`src/<module>/`** — mỗi module một thư mục, có `__init__.py`.
- **`configs/`** — single source of truth cho hyperparameter, dùng YAML cho dễ đọc.
- **`tests/`** — tách unit (nhanh) và integration (cần data).

## `.gitignore` mẫu

```gitignore
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
.venv/
venv/
env/

# IDE
.vscode/
.idea/
*.swp

# Data
data/raw/
data/processed/
data/splits/

# Models
models/checkpoints/
models/trained/
*.pt
*.pth
*.onnx
*.h5

# Results
results/runs/*/checkpoints/
results/runs/*/*.pt
results/runs/*/*.pth

# Logs
*.log
logs/

# Secrets
.env
*.key
secrets.yaml
```

## Script tham khảo

Tạo script `src/<module>/__main__.py` để chạy module như một CLI:

```python
# src/data/__main__.py
import argparse
from src.data.dataset import build_dataloader

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    loader = build_dataloader(args.config)
    for batch in loader:
        print(batch["image"].shape, batch["mask"].shape)
        break

if __name__ == "__main__":
    main()
```

Chạy: `python -m src.data --config configs/data_config.yaml`.