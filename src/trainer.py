"""Vòng lặp huấn luyện cho một nhánh (mặt hoặc tiếng) với log chi tiết từng tham số."""
from __future__ import annotations

import contextlib
import math
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from . import data_utils as du
from .datasets import build_datasets
from .labels import class_names
from .logging_utils import RunLogger
from .metrics import by_group, plot_confusion, summarize
from .models import build_model
from .param_tracker import ParamTracker


# ----------------------------------------------------------------- tiện ích
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pick_device(name: str = "auto") -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def amp_context(enabled: bool):
    return torch.autocast(device_type="cuda", dtype=torch.float16) if enabled else contextlib.nullcontext()


def make_scaler(enabled: bool):
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


def load_ckpt(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:  # torch cũ không có weights_only
        return torch.load(path, map_location=device)


def make_loaders(data, cfg, device):
    t = cfg["train"]
    bs, nw = int(t["batch_size"]), int(t["num_workers"])
    g = torch.Generator()
    g.manual_seed(int(cfg["seed"]))
    common = dict(num_workers=nw, pin_memory=(device.type == "cuda"), persistent_workers=(nw > 0))
    loaders = {}
    if "train" in data["datasets"]:
        ds = data["datasets"]["train"]
        loaders["train"] = DataLoader(ds, batch_size=bs, shuffle=True, drop_last=len(ds) > bs,
                                      generator=g, **common)
    for sp in ("val", "test"):
        if sp in data["datasets"]:
            loaders[sp] = DataLoader(data["datasets"][sp], batch_size=bs * 2, shuffle=False, **common)
    return loaders


def make_optimizer(model, t):
    params = [p for p in model.parameters() if p.requires_grad]
    decay = [p for p in params if p.ndim > 1]       # trọng số conv/linear: có weight decay
    no_decay = [p for p in params if p.ndim <= 1]   # bias và BatchNorm: không decay
    groups = [{"params": decay, "weight_decay": float(t["weight_decay"])},
              {"params": no_decay, "weight_decay": 0.0}]
    lr = float(t["lr"])
    if t["optimizer"] == "adamw":
        return torch.optim.AdamW(groups, lr=lr)
    if t["optimizer"] == "adam":
        return torch.optim.Adam(groups, lr=lr)
    return torch.optim.SGD(groups, lr=lr, momentum=0.9, nesterov=True)


def make_scheduler(optimizer, t, steps_per_epoch: int):
    total = int(t["epochs"]) * steps_per_epoch
    warm = int(round(float(t["warmup_epochs"]) * steps_per_epoch))
    min_r = float(t["min_lr_ratio"])

    def factor(step):
        if warm > 0 and step < warm:
            return (step + 1) / warm
        prog = min(1.0, (step - warm) / max(1, total - warm))
        return min_r + (1.0 - min_r) * 0.5 * (1.0 + math.cos(math.pi * prog))

    return LambdaLR(optimizer, factor)


@torch.no_grad()
def predict(model, loader, device, use_amp=False):
    """Trả về (xác suất softmax [N,C], nhãn thật [N]) theo đúng thứ tự dataset."""
    model.eval()
    probs, labels = [], []
    for x, y, _ in loader:
        x = x.to(device, non_blocking=True)
        with amp_context(use_amp):
            logits = model(x)
        probs.append(torch.softmax(logits.float(), dim=1).cpu().numpy())
        labels.append(y.numpy())
    return np.concatenate(probs), np.concatenate(labels)


# ----------------------------------------------------------------- 1 epoch
def train_one_epoch(model, loader, criterion, optimizer, scheduler, scaler, device, cfg,
                    tracker, logger, epoch, global_step, use_amp, grad_hist):
    t, lg = cfg["train"], cfg["logging"]
    clip = float(t["grad_clip"])
    max_norm = clip if clip > 0 else float("inf")
    log_every = max(1, int(lg["log_every_steps"]))
    model.train()
    loss_sum = torch.zeros((), device=device)
    correct = torch.zeros((), device=device)
    gn_sum = torch.zeros((), device=device)
    gn_cnt = torch.zeros((), device=device)
    seen = 0
    n_batches = len(loader)
    t0 = time.time()
    for step, (x, y, _) in enumerate(loader):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with amp_context(use_amp):
            logits = model(x)
            loss = criterion(logits, y)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)                      # gradient về thang thật trước khi đo/cắt
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        tracker.sample_grads(global_step)
        if grad_hist and step == n_batches - 1:
            logger.log_grad_hists(model, epoch)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        bs = y.size(0)
        seen += bs
        loss_sum += loss.detach().float() * bs
        correct += (logits.detach().argmax(1) == y).sum()
        finite = torch.isfinite(gnorm)
        gn_sum += torch.where(finite, gnorm, torch.zeros_like(gnorm))
        gn_cnt += finite.float()
        if global_step % log_every == 0:
            logger.log_step({
                "global_step": global_step, "epoch": epoch, "step": step,
                "loss": float(loss.item()),
                "batch_acc": float((logits.argmax(1) == y).float().mean().item()),
                "lr": float(optimizer.param_groups[0]["lr"]),
                "grad_norm": float(gnorm.item()),
                "loss_scale": float(scaler.get_scale()) if scaler.is_enabled() else 1.0,
            })
        global_step += 1
    secs = time.time() - t0
    out = {
        "train_loss": float(loss_sum.item() / max(1, seen)),
        "train_acc": float(correct.item() / max(1, seen)),
        "grad_norm_mean": float(gn_sum.item() / max(1.0, gn_cnt.item())),
        "samples_per_sec": seen / max(secs, 1e-9),
    }
    return out, global_step


# ----------------------------------------------------------------- báo cáo cuối
def final_report(model, loaders, data, cfg, names, out_dir, device, use_amp, splits=("val", "test"), logger=None):
    """Đánh giá checkpoint tốt nhất; lưu xác suất, ma trận nhầm lẫn, chỉ số từng lớp (và theo nguồn)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    n = len(names)
    say = logger.info if logger else print
    res = {}
    for sp in splits:
        if sp not in loaders:
            continue
        P, Y = predict(model, loaders[sp], device, use_amp)
        s = summarize(Y, P, n)
        np.save(out_dir / f"probs_{sp}.npy", P.astype(np.float32))
        np.save(out_dir / f"labels_{sp}.npy", Y.astype(np.int64))
        meta = data["meta"].get(sp)
        item = {"n": int(len(Y)), "acc": s["acc"], "macro_f1": s["macro_f1"], "bal_acc": s["bal_acc"],
                "loss": s["loss"],
                "per_class": [{"class": names[i], "precision": float(s["precision"][i]),
                               "recall": float(s["recall"][i]), "f1": float(s["f1"][i]),
                               "support": int(s["support"][i])} for i in range(n)]}
        if meta is not None and "source" in meta.columns:
            src = meta["source"].astype(str).to_numpy()
            np.save(out_dir / f"sources_{sp}.npy", src)
            item["by_source"] = by_group(Y, P, src, n)
        pd.DataFrame(s["cm"], index=names, columns=names).to_csv(out_dir / f"confusion_{sp}.csv")
        plot_confusion(s["cm"], names, out_dir / f"confusion_{sp}.png",
                       title=f"{cfg['branch']} - {sp} (acc={s['acc']:.3f}, macro-F1={s['macro_f1']:.3f})")
        res[sp] = item
        say(f"[{sp}] acc={s['acc']:.4f} macro-F1={s['macro_f1']:.4f} bal-acc={s['bal_acc']:.4f} loss={s['loss']:.4f}")
        for pc in item["per_class"]:
            say(f"    {pc['class']:<9} P={pc['precision']:.3f} R={pc['recall']:.3f} F1={pc['f1']:.3f} n={pc['support']}")
        for g, v in item.get("by_source", {}).items():
            say(f"    nguồn {g:<8} n={v['n']:<5} acc={v['acc']:.3f} macro-F1={v['macro_f1']:.3f}")
    return res


# ----------------------------------------------------------------- fit
def fit(cfg: dict, run_dir, resume: bool = False, init_ckpt: str | None = None) -> dict:
    run_dir = Path(run_dir)
    logger = RunLogger(run_dir, cfg, use_tensorboard=bool(cfg["logging"]["tensorboard"]))
    try:
        return _fit(cfg, run_dir, logger, resume, init_ckpt=init_ckpt)
    finally:
        logger.close()


def _fit(cfg, run_dir, logger, resume, init_ckpt=None):
    t, lg = cfg["train"], cfg["logging"]
    n = int(cfg["num_classes"])
    names = class_names(n)
    set_seed(int(cfg["seed"]))
    device = pick_device(cfg["device"])
    use_amp = bool(t["amp"]) and device.type == "cuda"
    logger.info(f"Thiết bị: {device} | AMP: {use_amp} | nhánh: {cfg['branch']} | số lớp: {n}")
    logger.info("Cấu hình:\n" + yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True))

    data = build_datasets(cfg, log=logger.info)
    logger.write_json("data_summary.json", {"num_classes": n, "class_names": names, "counts": data["counts"],
                                            "mean": data["mean"], "std": data["std"]})
    loaders = make_loaders(data, cfg, device)

    model = build_model(cfg).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    (run_dir / "model.txt").write_text(f"{model}\n\nTổng tham số: {total_params:,} | huấn luyện được: {trainable:,}\n",
                                       encoding="utf-8")
    logger.info(f"Mô hình: {total_params:,} tham số ({trainable:,} huấn luyện được)")
    if cfg["branch"] == "face" and cfg["model"]["pretrained"] and not getattr(model, "pretrained_loaded", False):
        logger.info("CẢNH BÁO: không tải được trọng số ImageNet -> train từ khởi tạo ngẫu nhiên.")

    weights = du.class_weights(data["counts"]["train"], t["class_weight"])
    logger.info("Trọng số lớp (" + t["class_weight"] + "): "
                + ", ".join(f"{nm}={w:.2f}" for nm, w in zip(names, weights)))
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights, device=device),
                                    label_smoothing=float(t["label_smoothing"]))
    optimizer = make_optimizer(model, t)
    scheduler = make_scheduler(optimizer, t, len(loaders["train"]))
    scaler = make_scaler(use_amp)

    start_epoch, best, bad, global_step = 1, None, 0, 0
    last_path = run_dir / "last.pt"
    if resume and last_path.exists():
        ck = load_ckpt(last_path, device)
        model.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"])
        scheduler.load_state_dict(ck["scheduler"])
        scaler.load_state_dict(ck["scaler"])
        start_epoch, best, bad, global_step = ck["epoch"] + 1, ck["best"], ck["bad"], ck["global_step"]
        logger.info(f"Tiếp tục từ epoch {start_epoch} (đã có best = {best})")
    elif init_ckpt:
        ck = load_ckpt(init_ckpt, device)
        src_state = ck["model"] if isinstance(ck, dict) and "model" in ck else ck
        cur_state = model.state_dict()
        matched = {k: v for k, v in src_state.items() if k in cur_state and v.shape == cur_state[k].shape}
        skipped = [k for k in cur_state if k not in matched]
        cur_state.update(matched)
        model.load_state_dict(cur_state)
        logger.info(f"Fine-tune từ {init_ckpt}: nạp {len(matched)}/{len(cur_state)} tensor khớp kích thước"
                    + (f" (khởi tạo mới: {skipped})" if skipped else ""))
    tracker = ParamTracker(model, lg["param_every_steps"])

    metric = t["early_stop_metric"]
    mode_min = metric.endswith("loss")
    hist_every = int(lg["hist_every_epochs"])
    epochs = int(t["epochs"])
    patience = int(t["early_stop_patience"])

    for epoch in range(start_epoch, epochs + 1):
        t0 = time.time()
        grad_hist = hist_every > 0 and (epoch % hist_every == 0 or epoch == 1)
        tr, global_step = train_one_epoch(model, loaders["train"], criterion, optimizer, scheduler, scaler,
                                          device, cfg, tracker, logger, epoch, global_step, use_amp, grad_hist)
        P, Y = predict(model, loaders["val"], device, use_amp)
        s = summarize(Y, P, n)
        val = {"val_loss": s["loss"], "val_acc": s["acc"], "val_macro_f1": s["macro_f1"], "val_bal_acc": s["bal_acc"]}
        row = {"epoch": epoch, "lr": float(optimizer.param_groups[0]["lr"]), **tr, **val,
               "epoch_sec": time.time() - t0}
        logger.log_epoch(row)
        logger.log_per_class(epoch, names, s["precision"], s["recall"], s["f1"], s["support"])
        logger.log_params(tracker.epoch_rows(epoch))
        if grad_hist:
            logger.log_weight_hists(model, epoch)

        score = val[metric]
        improved = best is None or (score < best["score"] if mode_min else score > best["score"])
        if improved:
            best = {"score": float(score), "epoch": epoch, "val": val}
            bad = 0
            torch.save({"model": model.state_dict(), "cfg": cfg, "epoch": epoch, "val": val,
                        "mean": data["mean"], "std": data["std"], "class_names": names, "num_classes": n},
                       run_dir / "best.pt")
        else:
            bad += 1
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
                    "epoch": epoch, "best": best, "bad": bad, "global_step": global_step, "cfg": cfg},
                   last_path)
        logger.info(f"Epoch {epoch:03d}/{epochs} | lr {row['lr']:.2e} | train loss {tr['train_loss']:.4f} "
                    f"acc {tr['train_acc']:.4f} | val loss {val['val_loss']:.4f} acc {val['val_acc']:.4f} "
                    f"macroF1 {val['val_macro_f1']:.4f} | |grad| {tr['grad_norm_mean']:.3f} | "
                    f"{row['epoch_sec']:.1f}s{'  *tốt nhất*' if improved else ''}")
        if bad >= patience:
            logger.info(f"Dừng sớm: {patience} epoch liên tiếp không cải thiện {metric}.")
            break

    logger.info(f"Tải checkpoint tốt nhất (epoch {best['epoch']}, {metric}={best['score']:.4f}) để đánh giá cuối.")
    ck = load_ckpt(run_dir / "best.pt", device)
    model.load_state_dict(ck["model"])
    results = final_report(model, loaders, data, cfg, names, run_dir, device, use_amp, logger=logger)
    out = {"run_dir": str(run_dir), "branch": cfg["branch"], "num_classes": n, "class_names": names,
           "best_epoch": best["epoch"], "best_val": best["val"], "splits": results}
    logger.write_json("results.json", out)
    logger.info(f"Xong. Kết quả: {run_dir / 'results.json'}")
    return out
