"""CLI chính cho công cụ cào dữ liệu cảm xúc YouTube Tiếng Việt (Tích hợp Lip-Sync + Đa phương thức + Cào quy mô lớn 5.000 video / 10.000 mẫu)."""
from __future__ import annotations
import argparse
from pathlib import Path

from .downloader import iter_youtube_videos, parse_links_file
from .extractor import extract_face_audio_clips, ExtractedClip
from .pseudo_labeler import pseudo_label_clips, LabeledSample
from .review_ui import save_metadata_csv, build_review_html
from .export_splits import export_to_project_splits


def cmd_crawl(args):
    targets: list[str | tuple[str, str]] = []
    if args.urls:
        targets.extend(args.urls)
    if args.query:
        targets.append(f"ytsearch{args.max_videos}:{args.query}")

    urls_file = args.urls_file or ("tools/yt_crawler/youtube_links.txt" if not targets else None)
    if urls_file and Path(urls_file).exists():
        parsed = parse_links_file(urls_file)
        print(f"[yt_crawler] Đọc được {len(parsed)} mục từ file {urls_file}")
        targets.extend(parsed)

    if not targets:
        print("[yt_crawler] Vui lòng truyền --urls, --urls-file hoặc --query.")
        return

    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    csv_path = out_root / "metadata.csv"

    print(
        f"[yt_crawler] Bắt đầu luồng cào & lọc Lip-Sync (Mục tiêu tối đa: "
        f"{args.target_videos} video raw / {args.target_clips} mẫu processed)..."
    )

    all_labeled: list[LabeledSample] = []
    video_count = 0

    for v in iter_youtube_videos(
        targets,
        raw_dir=args.raw_dir,
        max_duration_sec=args.max_duration,
        max_total_videos=args.target_videos,
    ):
        video_count += 1
        clips = extract_face_audio_clips(
            v,
            out_dir=args.out_dir,
            clip_duration=args.clip_duration,
            step_sec=args.step_sec,
            max_clips_per_video=args.max_clips,
        )
        if clips:
            labeled_batch = pseudo_label_clips(clips, face_ckpt=args.face_ckpt, audio_ckpt=args.audio_ckpt)
            all_labeled.extend(labeled_batch)
            # Lưu tăng trưởng (incremental checkpoint) ngay sau mỗi video để không bao giờ mất dữ liệu
            save_metadata_csv(all_labeled, csv_path)
            print(
                f"  [{video_count}/{args.target_videos}] + Video {v.video_id} ({v.title[:36]}): "
                f"lấy được {len(clips)} mẫu khớp miệng nói (Tổng: {len(all_labeled)}/{args.target_clips} mẫu)"
            )
        else:
            print(f"  [{video_count}/{args.target_videos}] - Video {v.video_id}: không có đoạn nào khớp cử động miệng.")

        if args.cleanup_raw:
            try:
                v.video_path.unlink(missing_ok=True)
                v.video_path.with_suffix(".16k.wav").unlink(missing_ok=True)
            except Exception:
                pass

        if len(all_labeled) >= args.target_clips:
            print(f"[yt_crawler] Đã đạt mục tiêu {len(all_labeled)} mẫu processed! Dừng cào.")
            break

    if not all_labeled:
        print("[yt_crawler] Không tìm thấy đoạn nào có đồng thời khuôn mặt đang mở miệng nói và giọng nói.")
        return

    html_path = build_review_html(all_labeled, out_root / "review_labeler.html")
    print(f"[yt_crawler] Đã lưu {len(all_labeled)} mẫu tại: {csv_path}")
    print(f"[yt_crawler] Mở file HTML sau để nghe & kiểm tra nhãn: {html_path}")

    if len(all_labeled) >= 3:
        splits = export_to_project_splits(csv_path, splits_dir="data/splits", prefix="vn_")
        print("[yt_crawler] Đã sinh sẵn các file split + GroupKFold tại data/splits/:")
        for k, p in splits.items():
            print(f"  - {k}: {p}")


def cmd_split(args):
    splits = export_to_project_splits(args.csv, splits_dir=args.splits_dir, prefix=args.prefix)
    print("[yt_crawler] Đã cập nhật các file split từ CSV đã duyệt:")
    for k, p in splits.items():
        print(f"  - {k}: {p}")


def cmd_mock(args):
    """Tạo nhanh 6 mẫu giả lập tiếng Việt để kiểm tra giao diện HTML và pipeline fine-tune."""
    import numpy as np
    from PIL import Image
    import soundfile as sf

    out_root = Path(args.out_dir)
    face_dir = out_root / "faces"
    audio_dir = out_root / "audio"
    face_dir.mkdir(parents=True, exist_ok=True)
    audio_dir.mkdir(parents=True, exist_ok=True)

    mock_texts = [
        ("vid_vn01", "speaker_A", "Trời ơi tức thiệt sự luôn, sao làm ăn vô lý vậy!", 0.12),
        ("vid_vn01", "speaker_A", "Haha hôm nay vui quá mọi người ơi, cảm ơn rất nhiều!", 0.06),
        ("vid_vn02", "speaker_B", "Thực sự mình cảm thấy rất buồn và cô đơn lúc này...", 0.018),
        ("vid_vn02", "speaker_B", "Ủa thật á? Trời ơi bất ngờ quá không tin nổi luôn!", 0.09),
        ("vid_vn03", "speaker_C", "Hôm nay thời tiết Hà Nội khá mát mẻ và bình thường.", 0.03),
        ("vid_vn04", "speaker_D", "Nhìn chỗ đó dơ và kinh tởm quá không dám lại gần.", 0.035),
    ]

    sr = 16000
    t = np.linspace(0, 4.0, sr * 4, endpoint=False, dtype=np.float32)
    clips: list[ExtractedClip] = []

    for idx, (vid, spk, txt, amp) in enumerate(mock_texts):
        sid = f"{vid}_{idx:04d}"
        fp = face_dir / f"{sid}.png"
        ap = audio_dir / f"{sid}.wav"

        rng = np.random.default_rng(42 + idx)
        img_arr = rng.integers(60, 200, size=(48, 48), dtype=np.uint8)
        Image.fromarray(img_arr, mode="L").save(fp)

        freq = 220.0 + idx * 45.0
        wav = (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)
        sf.write(str(ap), wav, sr)

        clips.append(
            ExtractedClip(
                sample_id=sid,
                video_id=vid,
                speaker=spk,
                start_sec=idx * 4.0,
                end_sec=(idx + 1) * 4.0,
                face_path=fp.as_posix(),
                audio_path=ap.as_posix(),
                transcript_vi=txt,
                rms_energy=amp,
            )
        )

    labeled = pseudo_label_clips(clips)
    csv_path = save_metadata_csv(labeled, out_root / "metadata.csv")
    html_path = build_review_html(labeled, out_root / "review_labeler.html")
    splits = export_to_project_splits(csv_path, splits_dir="data/splits", prefix="vn_")

    print(f"[yt_crawler] Đã tạo {len(labeled)} mẫu thử nghiệm Tiếng Việt!")
    print(f"  1. Giao diện duyệt nhãn HTML: {html_path}")
    print(f"  2. File metadata CSV:         {csv_path}")
    print(f"  3. Các file splits để train:  {', '.join(str(p) for p in splits.values())}")


def main():
    parser = argparse.ArgumentParser(description="Công cụ cào & gán nhãn dữ liệu YouTube Tiếng Việt")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_crawl = sub.add_parser("crawl", help="Tải video YouTube, lọc người đang mở miệng nói (Lip-Sync) và gợi ý nhãn đa phương thức")
    p_crawl.add_argument("--urls", nargs="*", help="Danh sách link YouTube")
    p_crawl.add_argument("--urls-file", default=None, help="File .txt chứa danh sách link/từ khóa YouTube")
    p_crawl.add_argument("--query", default=None, help="Từ khóa tìm kiếm YouTube tiếng Việt")
    p_crawl.add_argument("--max-videos", type=int, default=10, help="Số video tối đa cho 1 câu lệnh --query lẻ")
    p_crawl.add_argument("--target-videos", type=int, default=5000, help="Tổng số video raw tối đa cần tải (mặc định 5000)")
    p_crawl.add_argument("--target-clips", type=int, default=10000, help="Tổng số mẫu processed tối đa cần trích xuất (mặc định 10000)")
    p_crawl.add_argument("--cleanup-raw", action="store_true", help="Tự động xóa file .mp4 gốc sau khi cắt xong để tiết kiệm ổ cứng khi cào 5000 video")
    p_crawl.add_argument("--max-duration", type=int, default=900, help="Bỏ qua video dài hơn số giây này (mặc định 900s = 15p)")
    p_crawl.add_argument("--clip-duration", type=float, default=4.0, help="Độ dài mỗi đoạn cắt (mặc định 4.0s)")
    p_crawl.add_argument("--step-sec", type=float, default=3.0, help="Bước trượt cửa sổ cắt (mặc định 3.0s)")
    p_crawl.add_argument("--max-clips", type=int, default=20, help="Số đoạn cắt tối đa mỗi video")
    p_crawl.add_argument("--raw-dir", default="data/raw/vn_youtube")
    p_crawl.add_argument("--out-dir", default="data/processed/vn_youtube")
    p_crawl.add_argument("--face-ckpt", default=None, help="Checkpoint mặt để gợi ý nhãn (nếu đã có)")
    p_crawl.add_argument("--audio-ckpt", default=None, help="Checkpoint tiếng để gợi ý nhãn (nếu đã có)")
    p_crawl.set_defaults(func=cmd_crawl)

    p_split = sub.add_parser("split", aliases=["export"], help="Chia train/val/test và GroupKFold từ file metadata.csv đã duyệt")
    p_split.add_argument("--csv", default="data/processed/vn_youtube/metadata.csv")
    p_split.add_argument("--splits-dir", default="data/splits")
    p_split.add_argument("--prefix", default="vn_")
    p_split.set_defaults(func=cmd_split)

    p_mock = sub.add_parser("mock", help="Sinh dữ liệu mẫu giả lập để test giao diện HTML & luồng fine-tune")
    p_mock.add_argument("--out-dir", default="data/processed/vn_youtube")
    p_mock.set_defaults(func=cmd_mock)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
