"""Vòng train chuẩn: config hóa, checkpoint best/last, early stopping, resume."""
from __future__ import annotations
import copy
import logging
import time
from pathlib import Path

import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.tensorboard import SummaryWriter

logger = logging.getLogger(__name__)


class Trainer:
    """Train 1 nhánh (face hoặc audio)."""

    def __init__(self, model: nn.Module, train_loader, val_loader,
                 lr: float = 3e-4, weight_decay: float = 1e-4,
                 epochs: int = 30, patience: int = 7,
                 grad_clip: float = 1.0, device: str | None = None,
                 ckpt_dir: str | Path = "models/checkpoints",
                 run_dir: str | Path | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.epochs = epochs
        self.patience = patience
        self.grad_clip = grad_clip
        self.ckpt_dir = Path(ckpt_dir)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.run_dir = Path(run_dir) if run_dir else None
        if self.run_dir:
            self.run_dir.mkdir(parents=True, exist_ok=True)
        self.opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        self.sched = torch.optim.lr_scheduler.CosineAnnealingLR(self.opt, T_max=epochs)
        self.scaler = torch.cuda.amp.GradScaler(enabled=(self.device == "cuda"))
        self.writer = SummaryWriter(log_dir=str(self.run_dir / "tb") if self.run_dir else None)
        self.best_f1 = -1.0
        self.bad_epochs = 0

    @torch.no_grad()
    def _predict(self, logits: torch.Tensor):
        return logits.argmax(dim=-1).cpu().tolist()

    def train_one_epoch(self, epoch: int) -> float:
        self.model.train()
        total, t0 = 0.0, time.time()
        for batch in self.train_loader:
            x = batch["image"] if "image" in batch else batch["mel"]
            y = batch["label"]
            x, y = x.to(self.device), y.to(self.device)
            self.opt.zero_grad()
            with torch.cuda.amp.autocast(enabled=(self.device == "cuda")):
                logits = self.model(x)
                loss = nn.functional.cross_entropy(logits, y)
            self.scaler.scale(loss).backward()
            if self.grad_clip:
                self.scaler.unscale_(self.opt)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.scaler.step(self.opt)
            self.scaler.update()
            total += float(loss) * len(y)
        self.sched.step()
        avg = total / max(1, len(self.train_loader.dataset))
        logger.info(f"epoch={epoch} train_loss={avg:.4f} lr={self.opt.param_groups[0]['lr']:.2e} ({time.time()-t0:.1f}s)")
        return avg

    @torch.no_grad()
    def validate(self) -> dict:
        self.model.eval()
        all_pred, all_true, total = [], [], 0.0
        for batch in self.val_loader:
            x = batch["image"] if "image" in batch else batch["mel"]
            y = batch["label"]
            x = x.to(self.device)
            logits = self.model(x)
            loss = nn.functional.cross_entropy(logits, y.to(self.device))
            total += float(loss) * len(y)
            all_pred += self._predict(logits)
            all_true += y.tolist()
        return {
            "loss": total / max(1, len(self.val_loader.dataset)),
            "acc": accuracy_score(all_true, all_pred),
            "f1_macro": f1_score(all_true, all_pred, average="macro", zero_division=0),
        }

    def save_checkpoint(self, epoch: int, name: str):
        path = self.ckpt_dir / name
        torch.save({
            "epoch": epoch,
            "model": self.model.state_dict(),
            "opt": self.opt.state_dict(),
            "sched": self.sched.state_dict(),
            "best_f1": self.best_f1,
        }, path)
        logger.info(f"saved {path}")

    def fit(self) -> Path:
        """Train full, early stopping theo val_f1_macro. Trả về đường dẫn best checkpoint."""
        best_path = self.ckpt_dir / "best.pt"
        for epoch in range(1, self.epochs + 1):
            tr_loss = self.train_one_epoch(epoch)
            m = self.validate()
            logger.info(f"epoch={epoch} val_loss={m['loss']:.4f} acc={m['acc']:.4f} f1={m['f1_macro']:.4f}")
            if self.run_dir:
                self.writer.add_scalar("loss/train", tr_loss, epoch)
                self.writer.add_scalar("loss/val", m["loss"], epoch)
                self.writer.add_scalar("f1/val", m["f1_macro"], epoch)
            self.save_checkpoint(epoch, "last.pt")
            if m["f1_macro"] > self.best_f1:
                self.best_f1 = m["f1_macro"]
                self.bad_epochs = 0
                # lưu best bằng copy state để tránh ghi đè race
                torch.save(copy.deepcopy(self.model.state_dict()), best_path.with_suffix(".state.pt"))
                self.save_checkpoint(epoch, "best.pt")
            else:
                self.bad_epochs += 1
                if self.bad_epochs >= self.patience:
                    logger.info(f"early stopping ở epoch {epoch} (best_f1={self.best_f1:.4f})")
                    break
        self.writer.close()
        return best_path
