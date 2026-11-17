"""Đọc cấu hình YAML, gộp với giá trị mặc định, và ghi đè bằng --set a.b=c."""
from __future__ import annotations

import copy
from pathlib import Path

import yaml

DEFAULTS = {
    "name": "run",
    "branch": "face",              # face | audio
    "data_root": ".",              # thư mục sao cho <data_root>/data/splits tồn tại
    "splits_dir": None,            # None = tự tìm <data_root>/data/splits
    "cache_dir": ".cache",         # nơi lưu bản đã parse (nhanh hơn từ lần 2)
    "num_classes": 7,              # 7 = cơ bản (0-6), 10 = đủ bộ
    "seed": 42,
    "device": "auto",              # auto | cpu | cuda | mps
    "model": {
        "pretrained": True,        # chỉ nhánh mặt: ResNet-18 ImageNet
        "remove_maxpool": False,   # chỉ nhánh mặt: bỏ maxpool đầu (ảnh nhỏ giữ chi tiết hơn)
        "input_size": 48,          # chỉ nhánh mặt: phóng ảnh lên cỡ này (vd 96)
        "dropout": 0.3,
    },
    "train": {
        "epochs": 60,
        "batch_size": 128,
        "optimizer": "adamw",      # adamw | adam | sgd
        "lr": 1e-3,
        "weight_decay": 1e-4,
        "warmup_epochs": 3,
        "min_lr_ratio": 0.01,
        "label_smoothing": 0.1,
        "class_weight": "inv_sqrt",  # none | inv | inv_sqrt
        "grad_clip": 1.0,          # <=0 là tắt
        "amp": True,               # chỉ có tác dụng trên GPU CUDA
        "num_workers": 2,          # Windows: nên đặt 0
        "early_stop_patience": 12,
        "early_stop_metric": "val_macro_f1",  # val_macro_f1 | val_acc | val_bal_acc | val_loss
    },
    "augment": {
        # mặt
        "hflip": True, "rotation": 10, "translate": 0.05, "erasing": 0.25,
        # tiếng (SpecAugment)
        "freq_mask": 16, "time_mask": 24, "n_masks": 2,
    },
    "logging": {
        "out_dir": "runs",
        "log_every_steps": 50,     # ghi loss/lr/grad mỗi N bước
        "param_every_steps": 10,   # lấy mẫu norm gradient từng tham số mỗi N bước
        "hist_every_epochs": 1,    # ghi histogram trọng số/gradient mỗi N epoch (0 = tắt)
        "tensorboard": True,
    },
}


def deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def apply_overrides(cfg: dict, items) -> dict:
    """items: ['train.epochs=5', 'model.pretrained=false', ...]"""
    cfg = copy.deepcopy(cfg)
    for item in items or []:
        if "=" not in item:
            raise ValueError(f"--set cần dạng khóa=giá_trị, nhận: {item!r}")
        key, raw = item.split("=", 1)
        value = yaml.safe_load(raw)
        node = cfg
        parts = key.strip().split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = value
    return cfg


def validate(cfg: dict) -> dict:
    if cfg["branch"] not in ("face", "audio"):
        raise ValueError("branch phải là 'face' hoặc 'audio'")
    if not 2 <= int(cfg["num_classes"]) <= 10:
        raise ValueError("num_classes phải trong khoảng 2..10")
    if cfg["train"]["class_weight"] not in ("none", "inv", "inv_sqrt"):
        raise ValueError("train.class_weight phải là none | inv | inv_sqrt")
    if cfg["train"]["optimizer"] not in ("adamw", "adam", "sgd"):
        raise ValueError("train.optimizer phải là adamw | adam | sgd")
    return cfg


def load_config(path, overrides=None) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        user = yaml.safe_load(f) or {}
    cfg = deep_merge(DEFAULTS, user)
    cfg = apply_overrides(cfg, overrides)
    return validate(cfg)


def save_config(cfg: dict, path) -> None:
    with open(Path(path), "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
