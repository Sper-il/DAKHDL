"""Theo dõi từng tensor tham số: nó thay đổi bao nhiêu sau mỗi epoch, gradient lớn cỡ nào."""
from __future__ import annotations

import torch


class ParamTracker:
    def __init__(self, model, every_steps: int = 10):
        pairs = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
        self.names = [n for n, _ in pairs]
        self.params = [p for _, p in pairs]
        self.prev = [p.detach().clone() for p in self.params]
        self.every = max(1, int(every_steps))
        self._reset()

    def _reset(self):
        dev = self.params[0].device
        self.gsum = torch.zeros(len(self.params), device=dev)
        self.gmax = torch.zeros(len(self.params), device=dev)
        self.gcount = 0

    @torch.no_grad()
    def sample_grads(self, step: int):
        """Gọi sau backward (và sau unscale) nhưng trước optimizer.step()."""
        if step % self.every:
            return
        grads = [p.grad for p in self.params]
        if any(g is None for g in grads):
            return
        try:
            norms = torch.stack(torch._foreach_norm(grads))
        except Exception:
            norms = torch.stack([g.norm() for g in grads])
        if not bool(torch.isfinite(norms).all()):
            return  # bước bị overflow (AMP) -> bỏ qua mẫu này
        self.gsum += norms
        self.gmax = torch.maximum(self.gmax, norms)
        self.gcount += 1

    @torch.no_grad()
    def epoch_rows(self, epoch: int):
        """Một dòng cho mỗi tensor tham số; cập nhật mốc so sánh cho epoch kế tiếp."""
        gmean = (self.gsum / max(1, self.gcount)).cpu().tolist()
        gmax = self.gmax.cpu().tolist()
        rows = []
        for i, (name, p) in enumerate(zip(self.names, self.params)):
            w = p.detach()
            prev = self.prev[i]
            w_norm = w.norm().item()
            d_norm = (w - prev).norm().item()
            rows.append({
                "epoch": epoch, "param": name, "shape": "x".join(map(str, w.shape)), "numel": w.numel(),
                "w_norm": w_norm, "w_mean": w.mean().item(),
                "w_std": w.std().item() if w.numel() > 1 else 0.0,
                "w_min": w.min().item(), "w_max": w.max().item(),
                "delta_norm": d_norm, "update_ratio": d_norm / (prev.norm().item() + 1e-12),
                "grad_norm_mean": gmean[i], "grad_norm_max": gmax[i], "grad_samples": self.gcount,
            })
            self.prev[i] = w.clone()
        self._reset()
        return rows
