"""Render report.html offline duy nhất từ metrics.json + plots/."""
from __future__ import annotations
import base64
import json
from pathlib import Path
from string import Template


def _img_b64(path: str | Path, max_side: int = 800) -> str:
    """Đọc ảnh, resize cạnh dài về max_side, trả về thẻ <img> base64."""
    from PIL import Image
    import io
    img = Image.open(path).convert("RGB")
    w, h = img.size
    scale = min(1.0, max_side / max(w, h))
    if scale < 1.0:
        img = img.resize((int(w * scale), int(h * scale)))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=80)
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f'<img src="data:image/jpeg;base64,{b64}" alt="{Path(path).name}">'


def metrics_table(metrics: dict) -> str:
    rows = []
    for name, info in metrics.items():
        if not isinstance(info, dict):
            continue
        acc = info.get("accuracy", "-")
        target = info.get("target", "-")
        passed = info.get("passed", None)
        cls = "pass" if passed else ("fail" if passed is False else "")
        mark = "ĐẠT" if passed else ("CHƯA ĐẠT" if passed is False else "-")
        rows.append(f"<tr><td>{name}</td><td>{target}</td><td>{acc}</td>"
                    f'<td class="{cls}">{mark}</td></tr>')
    return ("<table><tr><th>Nhánh</th><th>Target</th><th>Accuracy</th>"
            "<th>Kết luận</th></tr>" + "".join(rows) + "</table>")


def build_report(run_dir: str | Path, template_path: str | Path | None = None) -> str:
    """Render report.html trong run_dir. Trả về đường dẫn file HTML."""
    run_dir = Path(run_dir)
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    config = {}
    if (run_dir / "config.yaml").exists():
        config = (run_dir / "config.yaml").read_text(encoding="utf-8")[:2000]
    plots = run_dir / "plots"
    loss_img = next(plots.glob("*loss*.png"), None)
    cm_imgs = sorted(plots.glob("*confusion*.png")) + sorted(plots.glob("*cm*.png"))
    samples = sorted(plots.glob("sample*.jpg")) + sorted(plots.glob("sample*.png"))

    tpl_path = Path(template_path) if template_path else Path(
        ".cursor/skills/ai-fusion-workflow/templates/report.html")
    tpl = Template(tpl_path.read_text(encoding="utf-8"))
    html = tpl.safe_substitute(
        PROJECT="emotion-fusion",
        RUN_ID=run_dir.name,
        COMMIT=metrics.get("commit", "-"),
        TIME=metrics.get("time", "-"),
        CHECKPOINT=metrics.get("checkpoint", "-"),
        PROBLEM="Nhận diện 7 cảm xúc từ khuôn mặt và giọng nói (late fusion).",
        INPUT_DESC="Ảnh xám 48x48 + WAV 16 kHz 4 giây (mel 128x128)",
        OUTPUT_DESC="Nhãn cảm xúc + xác suất fusion 7 lớp + latency",
        METRICS_TABLE=metrics_table(metrics.get("branches", metrics)),
        LOSS_CURVE_IMG=_img_b64(loss_img) if loss_img else "<p>Chưa có đồ thị loss.</p>",
        DATA_DIST_IMG="<p>Xem phân bố lớp trong metrics.json.</p>",
        TRAIN_SAMPLES="".join(_img_b64(p) for p in samples[:6]) or "<p>Chưa có ảnh mẫu.</p>",
        GOOD_SAMPLES="".join(_img_b64(p) for p in samples[:6]) or "<p>Chưa có ảnh mẫu.</p>",
        BAD_CASES="<p>" + metrics.get("error_analysis", "Chưa có phân tích lỗi.") + "</p>",
        E2E_LATENCY_MS=metrics.get("e2e_latency_ms", "-"),
        THROUGHPUT_RPS=metrics.get("throughput_rps", "-"),
        VRAM_MB=metrics.get("vram_mb", "-"),
        LIMITATIONS="<ul><li>Chỉ ảnh xám 48x48, WAV 16 kHz mono.</li>"
                    "<li>Chưa nhiều người cùng ảnh, chưa video dài.</li></ul>",
        NEXT_STEPS="<ul><li>Học trọng số fusion trên val.</li>"
                   "<li>Thêm augment cho lớp hiếm.</li></ul>",
    )
    out = run_dir / "report.html"
    out.write_text(html, encoding="utf-8")
    return str(out)
