"""Vòng train chống Overfitting: Class Weights + Label Smoothing, Mixup, Model EMA (chuẩn BN buffer), Warmup-Cosine."""
from __future__ import annotations
import copy
import json
import logging
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.tensorboard import SummaryWriter

from src.models.architecture import build_loss, compute_smoothed_class_weights

logger = logging.getLogger(__name__)


class ModelEMA:
    """Exponential Moving Average của trọng số mô hình (làm mượt parameters, đồng bộ trực tiếp BatchNorm buffers)."""

    def __init__(self, model: nn.Module, decay: float = 0.99):
        self.ema = copy.deepcopy(model).eval()
        for p in self.ema.parameters():
            p.requires_grad_(False)
        self.decay = decay
        self.num_updates = 0

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        self.num_updates += 1
        d = min(self.decay, (1.0 + self.num_updates) / (10.0 + self.num_updates))
        model_params = dict(model.named_parameters())
        for name, ema_p in self.ema.named_parameters():
            if name in model_params:
                ema_p.mul_(d).add_(model_params[name].detach(), alpha=1.0 - d)
        # Đồng bộ trực tiếp BatchNorm running_mean / running_var để không bị lệch thống kê khi eval
        model_buffers = dict(model.named_buffers())
        for name, ema_b in self.ema.named_buffers():
            if name in model_buffers:
                ema_b.copy_(model_buffers[name].detach())


def _extract_labels_from_dataset(dataset) -> list[int] | None:
    """Lấy danh sách nhãn từ dataset để tính trọng số lớp cân bằng."""
    df = getattr(dataset, "df", None)
    if df is not None:
        col = "label" if "label" in df.columns else ("emotion" if "emotion" in df.columns else None)
        if col is not None:
            return df[col].astype(int).tolist()
    return None


class Trainer:
    """Train 1 nhánh (face hoặc audio) với bộ kỹ thuật chống overfitting toàn diện."""

    def __init__(
        self,
        model: nn.Module,
        train_loader,
        val_loader,
        lr: float = 3e-4,
        weight_decay: float = 1e-2,
        epochs: int = 30,
        patience: int = 8,
        grad_clip: float = 1.0,
        label_smoothing: float = 0.1,
        use_class_weights: bool = True,
        mixup_alpha: float = 0.2,
        mixup_prob: float = 0.4,
        use_ema: bool = True,
        ema_decay: float = 0.99,
        warmup_epochs: int = 2,
        freeze_epochs: int = 2,
        backbone_lr_mult: float = 0.35,
        device: str | None = None,
        ckpt_dir: str | Path = "models/checkpoints",
        run_dir: str | Path | None = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.epochs = epochs
        self.patience = patience
        self.grad_clip = grad_clip
        self.mixup_alpha = mixup_alpha
        self.mixup_prob = mixup_prob
        self.warmup_epochs = warmup_epochs
        self.freeze_epochs = freeze_epochs
        self.base_lr = lr
        self.weight_decay = weight_decay
        self.backbone_lr_mult = backbone_lr_mult

        self.ckpt_dir = Path(ckpt_dir)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.run_dir = Path(run_dir) if run_dir else None
        if self.run_dir:
            self.run_dir.mkdir(parents=True, exist_ok=True)

        class_weights = None
        if use_class_weights and train_loader is not None and hasattr(train_loader, "dataset"):
            labels = _extract_labels_from_dataset(train_loader.dataset)
            if labels:
                class_weights = compute_smoothed_class_weights(labels, num_classes=7, power=0.5).to(self.device)
        self.criterion = build_loss(class_weights=class_weights, label_smoothing=label_smoothing).to(self.device)
        self.val_criterion = nn.CrossEntropyLoss().to(self.device)

        if self.freeze_epochs > 0 and hasattr(self.model, "set_backbone_trainable"):
            self.model.set_backbone_trainable(False)

        self._build_optimizer_and_scheduler(remaining_epochs=epochs)

        self.use_amp = self.device == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)
        self.ema = ModelEMA(self.model, decay=ema_decay) if use_ema else None
        self.writer = SummaryWriter(log_dir=str(self.run_dir / "tb") if self.run_dir else None)
        self.best_f1 = -1.0
        self.bad_epochs = 0
        self.history: list[dict] = []

    def _build_optimizer_and_scheduler(self, remaining_epochs: int) -> None:
        if hasattr(self.model, "get_param_groups"):
            param_groups = self.model.get_param_groups(
                lr=self.base_lr,
                weight_decay=self.weight_decay,
                backbone_lr_mult=self.backbone_lr_mult,
            )
        else:
            param_groups = [{"params": [p for p in self.model.parameters() if p.requires_grad], "lr": self.base_lr}]

        self.opt = torch.optim.AdamW(param_groups, lr=self.base_lr, weight_decay=self.weight_decay)

        def lr_lambda(ep_idx: int) -> float:
            if self.warmup_epochs > 0 and ep_idx < self.warmup_epochs:
                return float(ep_idx + 1) / float(max(1, self.warmup_epochs))
            progress = float(ep_idx - self.warmup_epochs) / float(max(1, remaining_epochs - self.warmup_epochs))
            progress = min(1.0, max(0.0, progress))
            return 0.05 + 0.95 * 0.5 * (1.0 + math.cos(math.pi * progress))

        self.sched = torch.optim.lr_scheduler.LambdaLR(self.opt, lr_lambda=lr_lambda)

    @torch.no_grad()
    def _predict(self, logits: torch.Tensor):
        return logits.argmax(dim=-1).cpu().tolist()

    def train_one_epoch(self, epoch: int) -> tuple[float, float]:
        if self.freeze_epochs > 0 and epoch == self.freeze_epochs + 1 and hasattr(self.model, "set_backbone_trainable"):
            self.model.set_backbone_trainable(True)
            self._build_optimizer_and_scheduler(remaining_epochs=max(1, self.epochs - epoch + 1))
            logger.info(f"epoch={epoch} Đã mở khóa toàn bộ backbone với layer-wise LR (x{self.backbone_lr_mult})")

        self.model.train()
        total_loss, t0 = 0.0, time.time()
        all_pred, all_true = [], []

        for batch in self.train_loader:
            x = batch["image"] if "image" in batch else batch["mel"]
            y = batch["label"]
            x, y = x.to(self.device), y.to(self.device)
            self.opt.zero_grad()

            do_mixup = (
                self.mixup_alpha > 0.0
                and len(y) > 1
                and random.random() < self.mixup_prob
            )
            if do_mixup:
                lam = float(np.random.beta(self.mixup_alpha, self.mixup_alpha))
                perm = torch.randperm(x.size(0), device=self.device)
                x_in = lam * x + (1.0 - lam) * x[perm]
                y_a, y_b = y, y[perm]
            else:
                lam = 1.0
                x_in = x
                y_a = y_b = y

            with torch.amp.autocast("cuda", enabled=self.use_amp):
                logits = self.model(x_in)
                if do_mixup:
                    loss = lam * self.criterion(logits, y_a) + (1.0 - lam) * self.criterion(logits, y_b)
                else:
                    loss = self.criterion(logits, y)

            self.scaler.scale(loss).backward()
            if self.grad_clip:
                self.scaler.unscale_(self.opt)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.scaler.step(self.opt)
            self.scaler.update()

            if self.ema is not None:
                self.ema.update(self.model)

            total_loss += float(loss.item()) * len(y)
            all_pred += self._predict(logits.detach())
            all_true += y.cpu().tolist()

        self.sched.step()
        avg_loss = total_loss / max(1, len(self.train_loader.dataset))
        tr_acc = float(accuracy_score(all_true, all_pred)) if all_true else 0.0
        cur_lr = self.opt.param_groups[-1]["lr"]
        logger.info(
            f"epoch={epoch} train_loss={avg_loss:.4f} train_acc={tr_acc:.4f} lr={cur_lr:.2e} ({time.time()-t0:.1f}s)"
        )
        return avg_loss, tr_acc

    @torch.no_grad()
    def validate(self, use_ema_model: bool = True) -> dict:
        eval_model = self.ema.ema if (use_ema_model and self.ema is not None) else self.model
        eval_model.eval()
        all_pred, all_true, total = [], [], 0.0
        for batch in self.val_loader:
            x = batch["image"] if "image" in batch else batch["mel"]
            y = batch["label"]
            x = x.to(self.device)
            logits = eval_model(x)
            loss = self.val_criterion(logits, y.to(self.device))
            total += float(loss.item()) * len(y)
            all_pred += self._predict(logits)
            all_true += y.tolist()
        return {
            "loss": total / max(1, len(self.val_loader.dataset)),
            "acc": float(accuracy_score(all_true, all_pred)),
            "f1_macro": float(f1_score(all_true, all_pred, average="macro", zero_division=0)),
        }

    def save_checkpoint(self, epoch: int, name: str, use_ema_weights: bool = False):
        path = self.ckpt_dir / name
        target_model = self.ema.ema if (use_ema_weights and self.ema is not None) else self.model
        torch.save(
            {
                "epoch": epoch,
                "model": target_model.state_dict(),
                "opt": self.opt.state_dict(),
                "sched": self.sched.state_dict(),
                "best_f1": self.best_f1,
            },
            path,
        )
        logger.info(f"saved {path}")

    def _save_training_plots(self) -> None:
        if not self.run_dir or not self.history:
            return
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            plots_dir = self.run_dir / "plots"
            plots_dir.mkdir(parents=True, exist_ok=True)
            ep = [h["epoch"] for h in self.history]
            tr_l = [h["train_loss"] for h in self.history]
            va_l = [h["val_loss"] for h in self.history]
            va_f1 = [h["val_f1_macro"] for h in self.history]
            tr_acc = [h["train_acc"] for h in self.history]
            va_acc = [h["val_acc"] for h in self.history]

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
            ax1.plot(ep, tr_l, label="Train Loss", marker="o", markersize=3)
            ax1.plot(ep, va_l, label="Val Loss", marker="s", markersize=3)
            ax1.set_title("Loss Curve (Train vs Val)")
            ax1.set_xlabel("Epoch")
            ax1.set_ylabel("Loss")
            ax1.grid(True, alpha=0.3)
            ax1.legend()

            ax2.plot(ep, tr_acc, label="Train Acc", linestyle="--", alpha=0.7)
            ax2.plot(ep, va_acc, label="Val Acc", marker="o", markersize=3)
            ax2.plot(ep, va_f1, label="Val F1-Macro", marker="^", markersize=4)
            ax2.set_title("Accuracy & F1-Macro (Overfit Monitor)")
            ax2.set_xlabel("Epoch")
            ax2.set_ylabel("Score")
            ax2.grid(True, alpha=0.3)
            ax2.legend()

            fig.tight_layout()
            fig.savefig(plots_dir / "loss_curve.png", dpi=130)
            plt.close(fig)
            (self.run_dir / "history.json").write_text(json.dumps(self.history, indent=2), encoding="utf-8")
        except Exception as e:
            logger.warning(f"Không vẽ được đồ thị loss_curve.png: {e}")

    def fit(self) -> Path:
        """Train full, early stopping theo val_f1_macro + 0.5*val_acc. Trả về đường dẫn best checkpoint."""
        best_path = self.ckpt_dir / "best.pt"
        best_composite = -1.0
        for epoch in range(1, self.epochs + 1):
            tr_loss, tr_acc = self.train_one_epoch(epoch)
            m_raw = self.validate(use_ema_model=False)
            if self.ema is not None and epoch >= 4:
                m_ema = self.validate(use_ema_model=True)
                use_ema_now = (m_ema["f1_macro"] + 0.5 * m_ema["acc"]) >= (m_raw["f1_macro"] + 0.5 * m_raw["acc"])
                m = m_ema if use_ema_now else m_raw
            else:
                use_ema_now = False
                m = m_raw

            gap = tr_acc - m["acc"]
            logger.info(
                f"epoch={epoch} val_loss={m['loss']:.4f} acc={m['acc']:.4f} f1={m['f1_macro']:.4f} gap={gap:+.4f}"
            )
            if gap > 0.20 and epoch > 4:
                logger.warning(f"[chống overfit] Cảnh báo: train_acc - val_acc = {gap:.3f} (> 0.20)")

            self.history.append({
                "epoch": epoch,
                "train_loss": round(tr_loss, 4),
                "train_acc": round(tr_acc, 4),
                "val_loss": round(m["loss"], 4),
                "val_acc": round(m["acc"], 4),
                "val_f1_macro": round(m["f1_macro"], 4),
                "overfit_gap": round(gap, 4),
            })

            if self.run_dir:
                self.writer.add_scalar("loss/train", tr_loss, epoch)
                self.writer.add_scalar("loss/val", m["loss"], epoch)
                self.writer.add_scalar("f1/val", m["f1_macro"], epoch)
                self.writer.add_scalar("acc/gap", gap, epoch)

            self.save_checkpoint(epoch, "last.pt", use_ema_weights=use_ema_now)
            composite = m["f1_macro"] + 0.35 * m["acc"]
            if composite > best_composite:
                best_composite = composite
                self.best_f1 = m["f1_macro"]
                self.bad_epochs = 0
                target_state = (self.ema.ema if use_ema_now else self.model).state_dict()
                torch.save(copy.deepcopy(target_state), best_path.with_suffix(".state.pt"))
                self.save_checkpoint(epoch, "best.pt", use_ema_weights=use_ema_now)
            else:
                self.bad_epochs += 1
                if self.bad_epochs >= self.patience:
                    logger.info(f"early stopping ở epoch {epoch} (best_f1={self.best_f1:.4f})")
                    break

        self._save_training_plots()
        self.writer.close()
        return best_path
