"""Ghi log chi tiết: train.log (văn bản), các file CSV, và TensorBoard.

Các file trong thư mục run:
  train.log            nhật ký văn bản có dấu thời gian
  config.yaml          cấu hình đầy đủ đã dùng
  env.json             phiên bản Python/Torch/CUDA, tên GPU
  data_summary.json    số mẫu từng lớp ở mỗi split, mean/std chuẩn hóa
  model.txt            cấu trúc mô hình + số tham số
  steps.csv            mỗi N bước: loss, acc, lr, norm gradient tổng, loss scale
  epochs.csv           mỗi epoch: loss/acc/macro-F1 của train & val, lr, thời gian
  per_class.csv        mỗi epoch: precision/recall/F1/support từng lớp (val)
  params.csv           mỗi epoch, mỗi tensor tham số: norm/mean/std/min/max, mức thay đổi so với
                       epoch trước (update_ratio), norm gradient trung bình & lớn nhất
  tb/                  TensorBoard (scalar + histogram trọng số/gradient)
"""
from __future__ import annotations

import csv
import json
import logging
import platform
import sys
from pathlib import Path

from .config import save_config


class CsvLog:
    def __init__(self, path):
        self.path = Path(path)
        self.fields = None
        if self.path.exists() and self.path.stat().st_size > 0:  # tiếp tục run cũ (resume)
            with open(self.path, "r", encoding="utf-8") as f:
                self.fields = next(csv.reader(f))

    def write(self, row: dict):
        new = self.fields is None
        if new:
            self.fields = list(row.keys())
        with open(self.path, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f, lineterminator="\n")
            if new:
                w.writerow(self.fields)
            w.writerow([row.get(k, "") for k in self.fields])


class RunLogger:
    def __init__(self, run_dir, cfg: dict, use_tensorboard: bool = True):
        self.dir = Path(run_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.cfg = cfg
        self.log = logging.getLogger(f"emotion.{self.dir.resolve()}")
        self.log.setLevel(logging.INFO)
        self.log.propagate = False
        if not self.log.handlers:
            fmt = logging.Formatter("%(asctime)s | %(message)s", "%Y-%m-%d %H:%M:%S")
            fh = logging.FileHandler(self.dir / "train.log", encoding="utf-8")
            fh.setFormatter(fmt)
            sh = logging.StreamHandler(sys.stdout)
            sh.setFormatter(fmt)
            self.log.addHandler(fh)
            self.log.addHandler(sh)
        self.steps = CsvLog(self.dir / "steps.csv")
        self.epochs = CsvLog(self.dir / "epochs.csv")
        self.per_class = CsvLog(self.dir / "per_class.csv")
        self.params = CsvLog(self.dir / "params.csv")
        self.writer = None
        if use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self.writer = SummaryWriter(str(self.dir / "tb"))
            except Exception as e:
                self.info(f"TensorBoard không dùng được ({type(e).__name__}: {e}); vẫn ghi CSV.")
        save_config(cfg, self.dir / "config.yaml")
        self._write_env()

    # ------------------------------------------------------------- tiện ích
    def info(self, msg: str):
        self.log.info(msg)

    def write_json(self, name: str, obj):
        with open(self.dir / name, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2, ensure_ascii=False)

    def _write_env(self):
        env = {"python": sys.version.split()[0], "platform": platform.platform()}
        try:
            import numpy, pandas
            env["numpy"], env["pandas"] = numpy.__version__, pandas.__version__
        except Exception:
            pass
        try:
            import torch, torchvision
            env["torch"], env["torchvision"] = torch.__version__, torchvision.__version__
            env["cuda_available"] = bool(torch.cuda.is_available())
            if torch.cuda.is_available():
                env["gpu"] = torch.cuda.get_device_name(0)
                env["cuda"] = torch.version.cuda
        except Exception:
            pass
        self.write_json("env.json", env)

    # ------------------------------------------------------------- ghi log
    def log_step(self, row: dict):
        self.steps.write(row)
        if self.writer is not None:
            s = int(row["global_step"])
            for k, v in row.items():
                if k not in ("global_step", "epoch", "step") and isinstance(v, (int, float)):
                    self.writer.add_scalar(f"step/{k}", v, s)

    def log_epoch(self, row: dict):
        self.epochs.write(row)
        if self.writer is not None:
            e = int(row["epoch"])
            for k, v in row.items():
                if k != "epoch" and isinstance(v, (int, float)):
                    self.writer.add_scalar(f"epoch/{k}", v, e)

    def log_per_class(self, epoch: int, names, prec, rec, f1, support):
        for i, nm in enumerate(names):
            self.per_class.write({"epoch": epoch, "class_id": i, "class": nm,
                                  "precision": float(prec[i]), "recall": float(rec[i]),
                                  "f1": float(f1[i]), "support": int(support[i])})
            if self.writer is not None:
                self.writer.add_scalar(f"val_f1/{nm}", float(f1[i]), epoch)
                self.writer.add_scalar(f"val_recall/{nm}", float(rec[i]), epoch)

    def log_params(self, rows):
        for r in rows:
            self.params.write(r)
            if self.writer is not None:
                tag = r["param"].replace(".", "/")
                e = int(r["epoch"])
                self.writer.add_scalar(f"w_norm/{tag}", r["w_norm"], e)
                self.writer.add_scalar(f"update_ratio/{tag}", r["update_ratio"], e)
                self.writer.add_scalar(f"grad_norm/{tag}", r["grad_norm_mean"], e)

    def log_weight_hists(self, model, epoch: int):
        if self.writer is None:
            return
        for n, p in model.named_parameters():
            self.writer.add_histogram(f"weights/{n.replace('.', '/')}", p.detach().float().cpu(), epoch)

    def log_grad_hists(self, model, epoch: int):
        if self.writer is None:
            return
        for n, p in model.named_parameters():
            if p.grad is not None:
                self.writer.add_histogram(f"grads/{n.replace('.', '/')}", p.grad.detach().float().cpu(), epoch)

    def close(self):
        if self.writer is not None:
            self.writer.flush()
            self.writer.close()
        for h in list(self.log.handlers):
            h.flush()
            h.close()
            self.log.removeHandler(h)
