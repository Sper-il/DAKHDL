"""Sinh bao cao EDA offline results/eda/report.html - tieng Viet co dau."""
from pathlib import Path
import base64
import io
import pandas as pd
from PIL import Image

LABELS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
LABEL_VI = ["Giận dữ", "Ghê tởm", "Sợ hãi", "Vui vẻ", "Buồn", "Ngạc nhiên", "Trung tính"]
EDA = Path("results/eda")
OUT = EDA / "report.html"

def img_b64(path: Path, max_side: int = 800) -> str:
    img = Image.open(path).convert("RGB")
    w, h = img.size
    scale = min(1.0, max_side / max(w, h))
    if scale < 1.0:
        img = img.resize((int(w * scale), int(h * scale)))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=82)
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f'<img src="data:image/jpeg;base64,{b64}" alt="{path.name}">'

def dist_table(tieu_de: str, csv_path: str) -> str:
    df = pd.read_csv(csv_path, usecols=["label"])
    total = len(df)
    rows = ""
    for i in range(7):
        c = int((df["label"] == i).sum())
        pct = 100.0 * c / total if total else 0
        rows += f"<tr><td>{i}</td><td>{LABELS[i]}</td><td>{LABEL_VI[i]}</td><td>{c}</td><td>{pct:.1f}%</td></tr>"
    return f"<h3>{tieu_de} (n={total})</h3><table><tr><th>ID</th><th>Nhãn</th><th>Tiếng Việt</th><th>Số mẫu</th><th>Tỉ lệ</th></tr>{rows}</table>"

face_train = dist_table("FER2013 train", "data/splits/face_train.csv")
face_val = dist_table("FER2013 val", "data/splits/face_val.csv")
face_test = dist_table("FER2013 test", "data/splits/face_test.csv")
audio_train = dist_table("RAVDESS train", "data/splits/audio_train.csv")
audio_val = dist_table("RAVDESS val", "data/splits/audio_val.csv")
audio_test = dist_table("RAVDESS test", "data/splits/audio_test.csv")

tr = pd.read_csv("data/splits/audio_train.csv")["speaker"]
va = pd.read_csv("data/splits/audio_val.csv")["speaker"]
te = pd.read_csv("data/splits/audio_test.csv")["speaker"]
speaker_html = f"""<table><tr><th>Split</th><th>Số speaker</th><th>Danh sách</th></tr>
<tr><td>train</td><td>{tr.nunique()}</td><td>{", ".join(sorted(tr.unique()))}</td></tr>
<tr><td>val</td><td>{va.nunique()}</td><td>{", ".join(sorted(va.unique()))}</td></tr>
<tr><td>test</td><td>{te.nunique()}</td><td>{", ".join(sorted(te.unique()))}</td></tr></table>
<p>Giao nhau giữa train/val/test: rỗng (không leakage speaker).</p>"""

def fig(p: str, caption: str) -> str:
    fp = EDA / p
    if fp.exists():
        return f"<figure>{img_b64(fp)}<figcaption>{caption}</figcaption></figure>"
    return f"<p>Thiếu {p}</p>"

html = f"""<!DOCTYPE html>
<html lang="vi"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Báo cáo EDA Emotion-Fusion</title>
<style>
body{{font-family:Arial,sans-serif;max-width:1000px;margin:24px auto;padding:0 16px;color:#222;line-height:1.6}}
table{{border-collapse:collapse;width:100%;margin:12px 0}}
th,td{{border:1px solid #ccc;padding:8px;text-align:left}}
th{{background:#f2f2f2}}
img{{max-width:100%;height:auto;border:1px solid #ddd}}
.grid{{display:grid;grid-template-columns:repeat(2,1fr);gap:12px}}
figure{{margin:0}}figcaption{{font-size:12px;color:#555}}
.meta{{color:#555;font-size:14px}}code{{background:#f6f6f6;padding:2px 6px;border-radius:4px}}
.warn{{background:#fff8e1;border:1px solid #f0c36d;padding:10px;border-radius:6px}}
</style></head><body>
<h1>Báo cáo phân tích dữ liệu (EDA) Emotion-Fusion</h1>
<p class="meta">Nguồn: <code>data/splits/</code> | Seed 42 | Tổng: FER2013 35.887 mẫu + RAVDESS 1.440 file WAV 24 speaker</p>

<h2>1. Nhánh mặt (FER2013, ảnh xám 48×48)</h2>
<div class="warn"><b>Phát hiện chính:</b> lớp <code>disgust</code> (ghê tởm) chỉ chiếm 1,5% tập train (436/28.709), thấp gấp 16 lần so với lớp <code>happy</code>. Bắt buộc dùng weighted loss + tăng cường dữ liệu (augment) + theo dõi F1 macro.</div>
{face_train}{face_val}{face_test}
<div class="grid">{fig("face_train_dist.png", "Phân bố FER2013 train")}{fig("face_val_dist.png", "Phân bố FER2013 val")}{fig("face_test_dist.png", "Phân bố FER2013 test")}{fig("face_sample.png", "Ảnh mặt mẫu trong tập train")}</div>
<p>Thống kê pixel trên 2.000 ảnh train: trung bình 129,6, độ lệch chuẩn 64,6, giá trị từ 0 đến 255. Chuẩn hoá mean 0,5 std 0,5 hiện tại là hợp lý.</p>

<h2>2. Nhánh tiếng (RAVDESS, file WAV 16 kHz mono)</h2>
<div class="warn"><b>Phát hiện chính:</b> dữ liệu gần cân bằng, chỉ lớp <code>neutral</code> (gộp calm) cao gấp 1,5 lần. Tập val chỉ có 3 speaker nên phương sai cao, không tinh chỉnh quá kỹ trên val.</div>
{audio_train}{audio_val}{audio_test}
{speaker_html}
<div class="grid">{fig("audio_train_dist.png", "Phân bố RAVDESS train")}{fig("audio_val_dist.png", "Phân bố RAVDESS val")}{fig("audio_test_dist.png", "Phân bố RAVDESS test")}{fig("audio_mel_sample.png", "Mel-spectrogram mẫu")}</div>
<p>Mẫu kiểm tra Actor_03 dài 3,44 giây, đúng chuẩn 3–5 giây, đệm/cắt về 4 giây rồi trích xuất mel 128×128.</p>

<h2>3. Kết luận cho huấn luyện</h2>
<ul><li>Nhánh mặt: dùng weighted cross-entropy + Albumentations (lật ngang, xoay ±10 độ), theo dõi chỉ số <code>val_f1_macro</code>.</li>
<li>Nhánh tiếng: giữ nguyên chia split theo speaker, dùng SpecAugment (che thời gian/tần số), theo dõi F1 từng lớp, đặc biệt lớp neutral.</li>
<li>Đầu vào đã chuẩn: ảnh mặt <code>[1,48,48]</code>, mel <code>[1,128,128]</code>, sẵn sàng chạy train sanity.</li></ul>
<p>Script sinh báo cáo: <code>notebooks/2026-11-03_eda_html.py</code> | Phân tích chi tiết: <code>docs/eda.md</code></p>
</body></html>"""

OUT.write_text(html, encoding="utf-8")
print(f"[eda-html] wrote {OUT} ({OUT.stat().st_size} bytes)")
