"""Module tải video, âm thanh và phụ đề tiếng Việt từ YouTube bằng yt-dlp (hỗ trợ Streaming 5.000+ video & cache đĩa)."""
from __future__ import annotations
from collections.abc import Iterator
import re
from dataclasses import dataclass
from pathlib import Path


@dataclass
class DownloadedVideo:
    video_id: str
    title: str
    channel_id: str
    video_path: Path
    subtitles: list[tuple[float, float, str]]  # (start_sec, end_sec, text)
    category_emotion: str = ""  # Gợi ý cảm xúc từ đề mục file .txt (nếu có)


def parse_links_file(txt_path: str | Path) -> list[tuple[str, str]]:
    """Đọc file .txt chứa các đề mục (# [happy] ...) và link/query YouTube.

    Trả về danh sách (url_hoac_query, category_emotion).
    """
    p = Path(txt_path)
    if not p.exists():
        return []
    current_emotion = ""
    emo_tag_re = re.compile(r"\[(angry|disgust|fear|happy|sad|surprise|neutral)\]", re.IGNORECASE)
    items: list[tuple[str, str]] = []

    for raw_line in p.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("#"):
            m = emo_tag_re.search(line)
            if m:
                current_emotion = m.group(1).lower()
            continue
        items.append((line, current_emotion))

    # Tách link trực tiếp (ưu tiên chạy trước) và xoay vòng (round-robin) các cụm tìm kiếm theo 7 cảm xúc
    # để bất kể người dùng cào 30 video hay 5.000 video thì dữ liệu 7 nhãn luôn cân bằng đồng đều!
    direct_items = [it for it in items if not it[0].lower().startswith("ytsearch")]
    search_by_emo: dict[str, list[tuple[str, str]]] = {}
    for it in items:
        if it[0].lower().startswith("ytsearch"):
            search_by_emo.setdefault(it[1] or "other", []).append(it)

    interleaved_search: list[tuple[str, str]] = []
    max_len = max((len(v) for v in search_by_emo.values()), default=0)
    emo_order = ["happy", "sad", "angry", "surprise", "fear", "disgust", "neutral", "other"]
    for i in range(max_len):
        for emo in emo_order:
            lst = search_by_emo.get(emo, [])
            if i < len(lst):
                interleaved_search.append(lst[i])

    return direct_items + interleaved_search


def _parse_vtt_time(ts: str) -> float:
    """Chuyển đổi timestamp 'HH:MM:SS.mmm' hoặc 'MM:SS.mmm' sang giây."""
    parts = ts.strip().split(":")
    if len(parts) == 3:
        h, m, s = parts
        return int(h) * 3600 + int(m) * 60 + float(s.replace(",", "."))
    if len(parts) == 2:
        m, s = parts
        return int(m) * 60 + float(s.replace(",", "."))
    return 0.0


def parse_vtt_file(vtt_path: Path) -> list[tuple[float, float, str]]:
    """Đọc file phụ đề .vtt tiếng Việt thành danh sách (start_sec, end_sec, text)."""
    if not vtt_path.exists():
        return []
    content = vtt_path.read_text(encoding="utf-8", errors="ignore")
    lines = content.splitlines()
    segments: list[tuple[float, float, str]] = []
    time_pattern = re.compile(r"(\d{1,2}:\d{2}(?::\d{2})?[.,]\d{3})\s*-->\s*(\d{1,2}:\d{2}(?::\d{2})?[.,]\d{3})")

    i = 0
    while i < len(lines):
        m = time_pattern.search(lines[i])
        if m:
            start_s = _parse_vtt_time(m.group(1))
            end_s = _parse_vtt_time(m.group(2))
            i += 1
            text_lines = []
            while i < len(lines) and lines[i].strip() and not time_pattern.search(lines[i]):
                clean_line = re.sub(r"<[^>]+>", "", lines[i]).strip()
                if clean_line and not clean_line.startswith("Kind:") and not clean_line.startswith("Language:"):
                    text_lines.append(clean_line)
                i += 1
            text = " ".join(text_lines).strip()
            if text and (not segments or segments[-1][2] != text):
                segments.append((start_s, end_s, text))
        else:
            i += 1
    return segments


def get_ffmpeg_binary() -> str:
    """Lấy đường dẫn ffmpeg từ hệ thống hoặc gói imageio-ffmpeg."""
    import shutil
    sys_ff = shutil.which("ffmpeg")
    if sys_ff:
        return sys_ff
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return "ffmpeg"


def _extract_direct_video_id(target: str) -> str | None:
    """Trích nhanh video_id nếu target là link YouTube trực tiếp."""
    m = re.search(r"(?:v=|youtu\.be/|shorts/)([A-Za-z0-9_-]{11})", target)
    return m.group(1) if m else None


def iter_youtube_videos(
    targets: list[str | tuple[str, str]],
    raw_dir: str | Path = "data/raw/vn_youtube",
    max_duration_sec: int = 900,
    max_total_videos: int = 5000,
) -> Iterator[DownloadedVideo]:
    """Generator tải và trả về từng video ngay khi tải xong (tận dụng cache đĩa nếu đã có file .mp4)."""
    raw_root = Path(raw_dir)
    raw_root.mkdir(parents=True, exist_ok=True)
    seen_ids: set[str] = set()
    yielded_count = 0

    try:
        import yt_dlp
    except ImportError as exc:
        raise RuntimeError("Chưa cài đặt yt-dlp. Hãy chạy: pip install yt-dlp") from exc

    ffmpeg_bin = get_ffmpeg_binary()
    ydl_opts = {
        "format": "134+140/bestvideo[height<=360][ext=mp4]+bestaudio[ext=m4a]/bestvideo[height<=480]+bestaudio/best",
        "merge_output_format": "mp4",
        "ffmpeg_location": ffmpeg_bin,
        "outtmpl": str(raw_root / "%(id)s" / "%(id)s.%(ext)s"),
        "writesubtitles": True,
        "writeautomaticsub": True,
        "subtitleslangs": ["vi"],
        "subtitlesformat": "vtt",
        "quiet": True,
        "no_warnings": True,
        "ignoreerrors": True,
        "match_filter": yt_dlp.utils.match_filter_func(f"duration <= {max_duration_sec}"),
    }

    # 1. Quét và trả về trước các video có link trực tiếp đã nằm sẵn trong cache ổ cứng
    remaining_targets: list[tuple[str, str]] = []
    for item in targets:
        if yielded_count >= max_total_videos:
            break
        target, cat_emo = item if isinstance(item, tuple) else (item, "")
        direct_vid = _extract_direct_video_id(target)
        if direct_vid and direct_vid not in seen_ids:
            vid_dir = raw_root / direct_vid
            cached_mp4 = vid_dir / f"{direct_vid}.mp4"
            if cached_mp4.exists() and cached_mp4.stat().st_size > 50_000:
                vtt_cands = list(vid_dir.glob("*.vtt"))
                subs = parse_vtt_file(vtt_cands[0]) if vtt_cands else []
                seen_ids.add(direct_vid)
                yielded_count += 1
                yield DownloadedVideo(
                    video_id=direct_vid,
                    title=direct_vid,
                    channel_id=f"yt_{direct_vid}",
                    video_path=cached_mp4,
                    subtitles=subs,
                    category_emotion=cat_emo,
                )
                continue
        remaining_targets.append((target, cat_emo))

    # 2. Quét toàn bộ các thư mục video khác đã tải sẵn trong data/raw/vn_youtube/ để tận dụng 100% cache đĩa
    for vid_dir in sorted(raw_root.iterdir()):
        if yielded_count >= max_total_videos:
            break
        if not vid_dir.is_dir():
            continue
        vid = vid_dir.name
        if vid in seen_ids:
            continue
        cached_mp4 = vid_dir / f"{vid}.mp4"
        if cached_mp4.exists() and cached_mp4.stat().st_size > 50_000:
            vtt_cands = list(vid_dir.glob("*.vtt"))
            subs = parse_vtt_file(vtt_cands[0]) if vtt_cands else []
            seen_ids.add(vid)
            yielded_count += 1
            yield DownloadedVideo(
                video_id=vid,
                title=vid,
                channel_id=f"yt_{vid}",
                video_path=cached_mp4,
                subtitles=subs,
                category_emotion="",
            )

    # 3. Tải mới từ YouTube đối với các truy vấn còn lại
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        for target, cat_emo in remaining_targets:
            if yielded_count >= max_total_videos:
                break

            try:
                info = ydl.extract_info(target, download=True)
            except Exception as e:
                print(f"  [!] Bỏ qua nguồn lỗi ({target}): {e}")
                continue
            if not info:
                continue

            entries = info.get("entries", [info])
            for entry in entries:
                if yielded_count >= max_total_videos:
                    break
                if not entry:
                    continue
                vid = entry.get("id")
                if not vid or vid in seen_ids:
                    continue

                vid_dir = raw_root / vid
                video_path = vid_dir / f"{vid}.mp4"
                if not video_path.exists():
                    cands = list(vid_dir.glob(f"{vid}.*"))
                    video_cands = [c for c in cands if c.suffix.lower() in {".mp4", ".mkv", ".webm"}]
                    if not video_cands:
                        continue
                    video_path = video_cands[0]

                vtt_cands = list(vid_dir.glob("*.vtt"))
                subs = parse_vtt_file(vtt_cands[0]) if vtt_cands else []
                channel = entry.get("channel_id") or entry.get("uploader_id") or f"yt_{vid}"
                title = entry.get("title") or vid
                seen_ids.add(vid)
                yielded_count += 1
                yield DownloadedVideo(
                    video_id=vid,
                    title=title,
                    channel_id=str(channel),
                    video_path=video_path,
                    subtitles=subs,
                    category_emotion=cat_emo,
                )


def download_youtube_videos(
    targets: list[str | tuple[str, str]],
    raw_dir: str | Path = "data/raw/vn_youtube",
    max_duration_sec: int = 900,
    max_total_videos: int = 5000,
) -> list[DownloadedVideo]:
    """Tải danh sách URL/từ khóa YouTube và trả về list DownloadedVideo."""
    return list(
        iter_youtube_videos(
            targets=targets,
            raw_dir=raw_dir,
            max_duration_sec=max_duration_sec,
            max_total_videos=max_total_videos,
        )
    )
