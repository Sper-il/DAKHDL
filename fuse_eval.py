#!/usr/bin/env python
"""Đánh giá late fusion mặt + tiếng từ xác suất đã lưu (chỉ cần numpy).

Vì ảnh mặt và âm thanh KHÔNG thuộc cùng một clip, ta ghép cặp ngẫu nhiên một mẫu mặt và một mẫu
tiếng CÙNG NHÃN (lặp nhiều seed). Trọng số fusion chọn trên val, báo cáo trên test.
  python fuse_eval.py --face-run runs/face_... --audio-run runs/audio_...
"""
import argparse
import csv
import datetime
import json
from pathlib import Path

import numpy as np

from src.labels import class_names
from src.metrics import summarize

NOTE = ("LƯU Ý HẠN CHẾ: mặt và tiếng đến từ các bộ data khác nhau nên cặp được ghép nhân tạo theo nhãn. "
        "Hai nhánh mắc lỗi ĐỘC LẬP, vì vậy lợi ích của fusion ở đây có thể cao hơn so với data thật "
        "(cùng một người, lỗi thường tương quan). Hãy ghi rõ điều này trong báo cáo.")


def load_run(run):
    d = Path(run)
    return {sp: (np.load(d / f"probs_{sp}.npy"), np.load(d / f"labels_{sp}.npy")) for sp in ("val", "test")}


def make_pairs(yf, ya, rng):
    """Với mỗi lớp có ở cả hai bên, lấy n = min(số mẫu mặt, số mẫu tiếng) mẫu không lặp ở mỗi bên."""
    fi, ai = [], []
    for c in np.intersect1d(np.unique(yf), np.unique(ya)):
        i, j = np.where(yf == c)[0], np.where(ya == c)[0]
        n = min(len(i), len(j))
        fi.append(rng.choice(i, n, replace=False))
        ai.append(rng.choice(j, n, replace=False))
    return np.concatenate(fi), np.concatenate(ai)


def fuse(pf, pa, w, method):
    if method == "arithmetic":
        return w * pf + (1 - w) * pa
    lp = w * np.log(np.clip(pf, 1e-9, 1)) + (1 - w) * np.log(np.clip(pa, 1e-9, 1))
    lp -= lp.max(1, keepdims=True)
    p = np.exp(lp)
    return p / p.sum(1, keepdims=True)


def evaluate(face, audio, weights, method, seeds, base_seed, n):
    """Trả về acc, f1 mỗi cái shape [len(weights), seeds] và f1 từng lớp [len(weights), seeds, n]."""
    (pf, yf), (pa, ya) = face, audio
    acc = np.zeros((len(weights), seeds))
    f1 = np.zeros((len(weights), seeds))
    f1c = np.zeros((len(weights), seeds, n))
    for s in range(seeds):
        rng = np.random.default_rng(base_seed + s)
        fi, ai = make_pairs(yf, ya, rng)
        y = yf[fi]
        for k, w in enumerate(weights):
            r = summarize(y, fuse(pf[fi], pa[ai], w, method), n)
            acc[k, s], f1[k, s], f1c[k, s] = r["acc"], r["macro_f1"], r["f1"]
    return acc, f1, f1c


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--face-run", required=True)
    ap.add_argument("--audio-run", required=True)
    ap.add_argument("--method", choices=["arithmetic", "geometric"], default="arithmetic")
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", help="thư mục ghi kết quả (mặc định <audio-run>/../fusion_<thời gian>)")
    a = ap.parse_args()

    face, audio = load_run(a.face_run), load_run(a.audio_run)
    n = face["val"][0].shape[1]
    if audio["val"][0].shape[1] != n:
        raise SystemExit(f"Số lớp khác nhau: mặt {n}, tiếng {audio['val'][0].shape[1]}")
    names = class_names(n)
    ws = np.round(np.arange(0.0, 1.0001, 0.1), 2)  # w = trọng số của nhánh MẶT

    vacc, vf1, _ = evaluate(face["val"], audio["val"], ws, a.method, a.seeds, a.seed, n)
    best = int(np.argmax(vf1.mean(1)))
    w_best = float(ws[best])
    tacc, tf1, tf1c = evaluate(face["test"], audio["test"], ws, a.method, a.seeds, a.seed + 1000, n)

    def row(label, k):
        return {"setting": label, "w_face": float(ws[k]),
                "acc_mean": float(tacc[k].mean()), "acc_std": float(tacc[k].std()),
                "macro_f1_mean": float(tf1[k].mean()), "macro_f1_std": float(tf1[k].std()),
                "f1_per_class": {names[c]: float(tf1c[k, :, c].mean()) for c in range(n)}}

    k_face, k_audio, k_half = int(np.argmax(ws)), int(np.argmin(ws)), int(np.argmin(abs(ws - 0.5)))
    rows = [row("chỉ mặt", k_face), row("chỉ tiếng", k_audio), row("fusion w=0.5", k_half),
            row(f"fusion w={w_best} (chọn trên val)", best)]

    out = Path(a.out) if a.out else Path(a.audio_run).parent / f"fusion_{datetime.datetime.now():%Y%m%d_%H%M%S}"
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "fusion_curve.csv", "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f, lineterminator="\n")
        wr.writerow(["w_face", "val_macro_f1_mean", "val_macro_f1_std", "val_acc_mean",
                     "test_macro_f1_mean", "test_macro_f1_std", "test_acc_mean"])
        for k, w in enumerate(ws):
            wr.writerow([w, vf1[k].mean(), vf1[k].std(), vacc[k].mean(), tf1[k].mean(), tf1[k].std(), tacc[k].mean()])
    result = {"method": a.method, "seeds": a.seeds, "w_face_best_on_val": w_best, "rows": rows, "note": NOTE}
    with open(out / "fusion_results.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"\nFusion ({a.method}), {a.seeds} lần ghép cặp ngẫu nhiên, test:")
    print(f"{'cấu hình':<34}{'acc':>16}{'macro-F1':>18}")
    for r in rows:
        print(f"{r['setting']:<34}{r['acc_mean']:>9.4f}±{r['acc_std']:.4f}{r['macro_f1_mean']:>11.4f}±{r['macro_f1_std']:.4f}")
    print("\n" + NOTE)
    print(f"Đã lưu: {out}")


if __name__ == "__main__":
    main()
