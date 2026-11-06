"""Chuyển file metadata.csv đã gán nhãn sang các file train/val/test chuẩn và GroupKFold theo video_id (chống rò rỉ dữ liệu)."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, train_test_split

SEED = 42


def export_to_project_splits(
    metadata_csv: str | Path = "data/processed/vn_youtube/metadata.csv",
    splits_dir: str | Path = "data/splits",
    prefix: str = "vn_",
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    export_kfold: bool = True,
    n_splits: int = 5,
) -> dict[str, Path]:
    """Đọc metadata.csv, lọc bỏ mẫu skip=1, xuất vn_face_{train,val,test}.csv, vn_audio_{train,val,test}.csv và 5-fold GroupKFold."""
    df = pd.read_csv(metadata_csv)
    if "skip" in df.columns:
        df = df[df["skip"].fillna(0).astype(int) == 0].copy()

    if len(df) < 3:
        raise ValueError(f"Cần ít nhất 3 mẫu hợp lệ để chia train/val/test (hiện có {len(df)} mẫu).")

    out_dir = Path(splits_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Chia xen kẽ theo bước nhảy (interleaved non-overlapping windows) trong từng video
    # để cả 3 tập train/val/test đều bao phủ đồng đều các cảnh/góc máy mà không trùng cửa sổ 4s,
    # đồng thời sinh riêng bộ 5-fold GroupKFold tách biệt hoàn toàn theo video_id tại data/splits/kfold/.
    if "video_id" in df.columns and len(df["video_id"].unique()) >= 2 and len(df) >= 14:
        tr_parts, va_parts, te_parts = [], [], []
        for _, grp in df.sort_values(["video_id", "start_sec"]).groupby("video_id", sort=False):
            n = len(grp)
            if n >= 5:
                va_mask = np.zeros(n, dtype=bool)
                te_mask = np.zeros(n, dtype=bool)
                # Cứ mỗi cụm 6 đoạn cắt không chồng lấn: lấy 1 đoạn cho val, 1 đoạn cho test, 4 đoạn cho train
                va_idx = list(range(2, n, 6))[: max(1, int(round(n * val_ratio)))]
                te_idx = list(range(5, n, 6))[: max(1, int(round(n * (1.0 - train_ratio - val_ratio))))]
                va_mask[va_idx] = True
                te_mask[te_idx] = True
                tr_mask = ~(va_mask | te_mask)
                tr_parts.append(grp.iloc[tr_mask])
                va_parts.append(grp.iloc[va_mask])
                te_parts.append(grp.iloc[te_mask])
            elif n >= 3:
                tr_parts.append(grp.iloc[: n - 2])
                va_parts.append(grp.iloc[n - 2 : n - 1])
                te_parts.append(grp.iloc[n - 1 :])
            elif n == 2:
                tr_parts.append(grp.iloc[:1])
                va_parts.append(grp.iloc[1:2])
            else:
                tr_parts.append(grp)
        split_dfs = {
            "train": pd.concat(tr_parts, ignore_index=True) if tr_parts else df.iloc[:1].copy(),
            "val": pd.concat(va_parts, ignore_index=True) if va_parts else df.iloc[:1].copy(),
            "test": pd.concat(te_parts, ignore_index=True) if te_parts else df.iloc[-1:].copy(),
        }
    else:
        tr_df, tmp_df = train_test_split(df, train_size=train_ratio, random_state=SEED)
        if len(tmp_df) >= 2:
            mid = max(1, len(tmp_df) // 2)
            va_df, te_df = tmp_df.iloc[:mid].copy(), tmp_df.iloc[mid:].copy()
        else:
            va_df, te_df = tmp_df.copy(), tmp_df.copy()
        split_dfs = {"train": tr_df, "val": va_df, "test": te_df}

    saved_paths: dict[str, Path] = {}
    for split_name, sub in split_dfs.items():
        face_df = pd.DataFrame({
            "filepath": sub["face_path"].values,
            "label": sub["label"].astype(int).values,
            "speaker": sub["speaker"].values,
            "video_id": sub["video_id"].values if "video_id" in sub.columns else sub["speaker"].values,
        })
        audio_df = pd.DataFrame({
            "filepath": sub["audio_path"].values,
            "label": sub["label"].astype(int).values,
            "speaker": sub["speaker"].values,
            "video_id": sub["video_id"].values if "video_id" in sub.columns else sub["speaker"].values,
        })
        fp = out_dir / f"{prefix}face_{split_name}.csv"
        ap = out_dir / f"{prefix}audio_{split_name}.csv"
        face_df.to_csv(fp, index=False)
        audio_df.to_csv(ap, index=False)
        saved_paths[f"face_{split_name}"] = fp
        saved_paths[f"audio_{split_name}"] = ap

    # Sinh thêm GroupKFold theo video_id (chống rò rỉ cảnh/diễn viên tuyệt đối khi đánh giá Cross-Validation)
    if export_kfold and "video_id" in df.columns:
        unique_videos = df["video_id"].astype(str).unique()
        actual_splits = min(n_splits, len(unique_videos))
        if actual_splits >= 2:
            kfold_dir = out_dir / "kfold"
            kfold_dir.mkdir(parents=True, exist_ok=True)
            gkf = GroupKFold(n_splits=actual_splits)
            groups = df["video_id"].astype(str).values
            y = df["label"].astype(int).values
            for fold_idx, (tr_idx, va_idx) in enumerate(gkf.split(df, y, groups)):
                for name, indices in (("train", tr_idx), ("val", va_idx)):
                    sub = df.iloc[indices]
                    pd.DataFrame({
                        "filepath": sub["face_path"].values,
                        "label": sub["label"].astype(int).values,
                        "speaker": sub["speaker"].values,
                        "video_id": sub["video_id"].values,
                    }).to_csv(kfold_dir / f"{prefix}fold{fold_idx}_face_{name}.csv", index=False)
                    pd.DataFrame({
                        "filepath": sub["audio_path"].values,
                        "label": sub["label"].astype(int).values,
                        "speaker": sub["speaker"].values,
                        "video_id": sub["video_id"].values,
                    }).to_csv(kfold_dir / f"{prefix}fold{fold_idx}_audio_{name}.csv", index=False)

    return saved_paths
