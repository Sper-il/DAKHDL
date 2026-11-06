"""Sinh file CSV gốc và giao diện HTML offline để kiểm tra/sửa nhãn nhanh."""
from __future__ import annotations
from dataclasses import asdict
import json
from pathlib import Path
import pandas as pd

from .pseudo_labeler import LabeledSample, EMOTIONS


def save_metadata_csv(samples: list[LabeledSample], csv_path: str | Path) -> Path:
    """Lưu danh sách mẫu đã gợi ý nhãn ra file CSV."""
    out = Path(csv_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = [asdict(s) for s in samples]
    df = pd.DataFrame(rows)
    df.to_csv(out, index=False, encoding="utf-8")
    return out


def build_review_html(samples: list[LabeledSample], html_path: str | Path) -> Path:
    """Sinh trang HTML tương tác offline để nghe tiếng Việt, xem mặt 48x48 và chỉnh nhãn."""
    out = Path(html_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    # Chuẩn hóa đường dẫn tương đối tính từ thư mục chứa file HTML
    items = []
    for s in samples:
        d = asdict(s)
        try:
            d["rel_face"] = Path(s.face_path).relative_to(out.parent).as_posix()
        except ValueError:
            d["rel_face"] = f"faces/{Path(s.face_path).name}"
        try:
            d["rel_audio"] = Path(s.audio_path).relative_to(out.parent).as_posix()
        except ValueError:
            d["rel_audio"] = f"audio/{Path(s.audio_path).name}"
        items.append(d)

    data_json = json.dumps(items, ensure_ascii=False)
    emotions_json = json.dumps(EMOTIONS, ensure_ascii=False)

    html = f"""<!DOCTYPE html>
<html lang="vi">
<head>
<meta charset="utf-8">
<title>Duyệt & Gán Nhãn Cảm Xúc Tiếng Việt (YouTube Crawler)</title>
<style>
  :root {{
    --bg: #0f172a; --card: #1e293b; --text: #f8fafc; --muted: #94a3b8;
    --accent: #38bdf8; --ok: #22c55e; --danger: #ef4444;
  }}
  body {{
    font-family: system-ui, -apple-system, sans-serif;
    background: var(--bg); color: var(--text); margin: 0; padding: 24px;
  }}
  .header {{
    display: flex; justify-content: space-between; align-items: center;
    background: var(--card); padding: 16px 24px; border-radius: 12px;
    position: sticky; top: 12px; z-index: 100; box-shadow: 0 8px 20px rgba(0,0,0,0.4);
    margin-bottom: 24px;
  }}
  .btn-export {{
    background: var(--ok); color: #000; font-weight: 700; border: none;
    padding: 12px 20px; border-radius: 8px; cursor: pointer; font-size: 15px;
  }}
  .btn-export:hover {{ opacity: 0.9; }}
  .grid {{
    display: grid; grid-template-columns: repeat(auto-fill, minmax(360px, 1fr)); gap: 16px;
  }}
  .card {{
    background: var(--card); border-radius: 12px; padding: 16px;
    border: 2px solid transparent; transition: 0.15s;
  }}
  .card.verified {{ border-color: var(--ok); }}
  .card.skipped {{ opacity: 0.45; border-color: var(--danger); }}
  .top-row {{ display: flex; gap: 16px; align-items: center; margin-bottom: 12px; }}
  .face-img {{
    width: 96px; height: 96px; image-rendering: pixelated;
    border-radius: 8px; border: 2px solid #475569; object-fit: cover;
  }}
  audio {{ width: 100%; height: 36px; margin: 6px 0; }}
  .transcript {{
    font-size: 14px; color: #e2e8f0; background: #0f172a;
    padding: 8px 10px; border-radius: 6px; margin-bottom: 10px; min-height: 20px;
  }}
  .meta {{ font-size: 12px; color: var(--muted); margin-bottom: 6px; }}
  .emo-btns {{ display: flex; flex-wrap: wrap; gap: 6px; margin-top: 8px; }}
  .emo-btn {{
    background: #334155; color: var(--text); border: 1px solid #475569;
    padding: 6px 10px; border-radius: 6px; cursor: pointer; font-size: 12px;
  }}
  .emo-btn.active {{
    background: var(--accent); color: #000; font-weight: 700; border-color: var(--accent);
  }}
  .skip-btn {{
    background: transparent; color: var(--danger); border: 1px solid var(--danger);
    padding: 6px 10px; border-radius: 6px; cursor: pointer; font-size: 12px; margin-left: auto;
  }}
  .skip-btn.active {{ background: var(--danger); color: #fff; }}
</style>
</head>
<body>
  <div class="header">
    <div>
      <h2 style="margin:0 0 4px 0;">🇻🇳 Công cụ duyệt & chỉnh nhãn cảm xúc YouTube Tiếng Việt</h2>
      <div id="stats" style="font-size:14px;color:var(--muted);">Đang tải...</div>
    </div>
    <button class="btn-export" onclick="exportCSV()">💾 Xuất file CSV đã duyệt (metadata.csv)</button>
  </div>

  <div class="grid" id="grid"></div>

<script>
const EMOTIONS = {emotions_json};
const VI_NAMES = {{
  angry: "0: Tức giận", disgust: "1: Ghê tởm", fear: "2: Sợ hãi",
  happy: "3: Vui vẻ", sad: "4: Buồn", surprise: "5: Ngạc nhiên", neutral: "6: Bình thường"
}};
let samples = {data_json};

function render() {{
  const grid = document.getElementById("grid");
  grid.innerHTML = "";
  let verifiedCount = 0, skipCount = 0;

  samples.forEach((s, idx) => {{
    if (s.verified) verifiedCount++;
    if (s.skip) skipCount++;

    const card = document.createElement("div");
    card.className = "card" + (s.skip ? " skipped" : (s.verified ? " verified" : ""));

    let btnsHtml = EMOTIONS.map((e, eIdx) => {{
      const isAct = (!s.skip && s.label === eIdx) ? "active" : "";
      return `<button class="emo-btn ${{isAct}}" onclick="setLabel(${{idx}}, ${{eIdx}})">${{VI_NAMES[e]}}</button>`;
    }}).join("");

    card.innerHTML = `
      <div class="top-row">
        <img class="face-img" src="${{s.rel_face}}" alt="face">
        <div style="flex:1">
          <div class="meta"><b>#${{s.sample_id}}</b> (${{s.start_sec}}s - ${{s.end_sec}}s) | Kênh: ${{s.speaker}}</div>
          <div class="meta">Gợi ý: <b style="color:#38bdf8">${{VI_NAMES[s.emotion]}}</b> (conf: ${{s.confidence}})</div>
          <div class="meta">Lý do: ${{s.reason}}</div>
        </div>
      </div>
      <audio controls preload="none" src="${{s.rel_audio}}"></audio>
      <div class="transcript">💬 ${{s.transcript_vi || "<i>(Không có phụ đề ở đoạn này)</i>"}}</div>
      <div class="emo-btns">
        ${{btnsHtml}}
        <button class="skip-btn ${{s.skip ? 'active' : ''}}" onclick="toggleSkip(${{idx}})">🗑️ Loại bỏ</button>
      </div>
    `;
    grid.appendChild(card);
  }});

  document.getElementById("stats").innerHTML =
    `Tổng số mẫu: <b>${{samples.length}}</b> | Đã duyệt tay: <b style="color:#22c55e">${{verifiedCount}}</b> | Loại bỏ: <b style="color:#ef4444">${{skipCount}}</b> | Sử dụng để train: <b>${{samples.length - skipCount}}</b>`;
}}

function setLabel(idx, labelIdx) {{
  samples[idx].label = labelIdx;
  samples[idx].emotion = EMOTIONS[labelIdx];
  samples[idx].verified = 1;
  samples[idx].skip = 0;
  render();
}}

function toggleSkip(idx) {{
  samples[idx].skip = samples[idx].skip ? 0 : 1;
  render();
}}

function escCSV(val) {{
  const s = String(val ?? "");
  if (s.includes(",") || s.includes('"') || s.includes("\\n")) {{
    return '"' + s.replace(/"/g, '""') + '"';
  }}
  return s;
}}

function exportCSV() {{
  const cols = ["sample_id","video_id","speaker","start_sec","end_sec","face_path","audio_path","transcript_vi","label","emotion","confidence","reason","verified","skip"];
  const lines = [cols.join(",")];
  samples.forEach(s => {{
    lines.push(cols.map(c => escCSV(s[c])).join(","));
  }});
  const blob = new Blob([lines.join("\\n")], {{type: "text/csv;charset=utf-8;"}});
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = "metadata.csv";
  a.click();
}}

render();
</script>
</body>
</html>
"""
    out.write_text(html, encoding="utf-8")
    return out

