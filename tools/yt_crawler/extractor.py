"""Module tách âm thanh 16kHz, căn chỉnh mặt 5-landmark, Nhận diện Cử động Miệng (Active Speaker Lip-Sync) & Sắc thái Mặt."""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
import subprocess
import urllib.request
import cv2
import numpy as np
from PIL import Image
import soundfile as sf

from .downloader import DownloadedVideo, get_ffmpeg_binary

YUNET_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/"
    "face_detection_yunet_2023mar.onnx"
)
YUNET_LOCAL_PATH = Path("tools/yt_crawler/face_detection_yunet_2023mar.onnx")

FER_ONNX_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/facial_expression_recognition/"
    "facial_expression_recognition_mobilefacenet_2022july.onnx"
)
FER_LOCAL_PATH = Path("tools/yt_crawler/fer_mobilefacenet.onnx")

# Chuẩn 5 điểm mốc khuôn mặt trên ảnh 112x112 (2 mắt, mũi, 2 khóe miệng)
STD_LANDMARKS_112 = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float32,
)


@dataclass
class ExtractedClip:
    sample_id: str
    video_id: str
    speaker: str
    start_sec: float
    end_sec: float
    face_path: str
    audio_path: str
    transcript_vi: str
    rms_energy: float
    category_emotion: str = ""
    lip_sync_score: float = 0.0
    face_expr_probs: list[float] = field(default_factory=list)


def ensure_yunet_model(model_path: Path = YUNET_LOCAL_PATH) -> Path:
    """Đảm bảo mô hình nhận diện khuôn mặt YuNet ONNX (227KB) đã có sẵn."""
    if not model_path.exists() or model_path.stat().st_size < 100_000:
        model_path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(YUNET_URL, str(model_path))
    return model_path


def ensure_fer_model(model_path: Path = FER_LOCAL_PATH) -> Path:
    """Đảm bảo mô hình nhận diện sắc thái khuôn mặt MobileFaceNet ONNX (4.8MB, 88.27% RAF-DB) đã có sẵn."""
    if not model_path.exists() or model_path.stat().st_size < 1_000_000:
        model_path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(FER_ONNX_URL, str(model_path))
    return model_path


def align_face_112(bgr_frame: np.ndarray, landmarks_5x2: np.ndarray) -> np.ndarray:
    """Căn chỉnh khuôn mặt về khung chuẩn 112x112 theo 5 điểm mốc chuẩn OpenCV Zoo FER."""
    src = landmarks_5x2.astype(np.float32)
    dst = STD_LANDMARKS_112
    tfm, _ = cv2.estimateAffinePartial2D(src, dst, method=cv2.LMEDS)
    if tfm is None:
        tfm, _ = cv2.estimateAffinePartial2D(src, dst)
    return cv2.warpAffine(bgr_frame, tfm, (112, 112), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def align_upper_face_gray_112(gray_frame: np.ndarray, landmarks_5x2: np.ndarray) -> np.ndarray:
    """Căn chỉnh mặt theo 3 điểm cứng phía trên (2 mắt + đỉnh mũi) để cố định đầu nhưng giữ nguyên cử động hàm/miệng."""
    src_upper = landmarks_5x2[:3].astype(np.float32)
    dst_upper = STD_LANDMARKS_112[:3]
    tfm, _ = cv2.estimateAffinePartial2D(src_upper, dst_upper)
    return cv2.warpAffine(gray_frame, tfm, (112, 112), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def infer_single_fer_probs_112(fer_net, bgr: np.ndarray) -> np.ndarray:
    """Chạy MobileFaceNet FER trên 1 ảnh mặt 112x112 đã căn chỉnh (kèm hiệu chỉnh Logit Adjustment RAF-DB)."""
    if fer_net is None:
        return np.ones(7, dtype=np.float32) / 7.0
    zoo_to_proj = [0, 1, 2, 3, 6, 4, 5]
    raf_priors = np.array([705.0, 717.0, 281.0, 4772.0, 2524.0, 1982.0, 1290.0], dtype=np.float32)
    raf_log_bias = np.log(raf_priors / raf_priors.mean())

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    rgb = (rgb - 0.5) / 0.5
    blob = cv2.dnn.blobFromImage(rgb)
    fer_net.setInput(blob, "data")
    logits = fer_net.forward(["label"])[0][0].astype(np.float32)
    logits = logits - 0.90 * raf_log_bias
    logits = logits - np.max(logits)
    exp_l = np.exp(logits / 1.25)
    p_zoo = exp_l / (np.sum(exp_l) + 1e-8)

    p_proj = np.zeros(7, dtype=np.float32)
    for z_idx, p_idx in enumerate(zoo_to_proj):
        p_proj[p_idx] = float(p_zoo[z_idx])
    return p_proj / (p_proj.sum() + 1e-8)


def select_peak_expressive_frame_and_probs(
    fer_net,
    track: list[tuple[int, float, float, np.ndarray, np.ndarray, np.ndarray, float, float]],
    category_emotion: str = "",
) -> tuple[np.ndarray, np.ndarray]:
    """Chọn khung hình có biểu cảm sắc thái rõ nhất trong chuỗi mặt người đang nói và tính phân bố FER."""
    emotions = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
    cat_idx = emotions.index(category_emotion) if category_emotion in emotions else -1

    frame_probs = [infer_single_fer_probs_112(fer_net, it[3]) for it in track]
    scores = []
    for it, p in zip(track, frame_probs):
        area_conf = it[7]
        if cat_idx == 6:
            expr_weight = 0.5 + float(p[6])
        elif 0 <= cat_idx < 6:
            # Ưu tiên khung hình bộc lộ cảm xúc rõ nhất (đặc biệt là cảm xúc phù hợp bối cảnh hoặc khác neutral)
            non_neutral = float(np.max(p[:6]))
            expr_weight = 0.4 + 0.8 * non_neutral + 0.8 * float(p[cat_idx])
        else:
            expr_weight = 0.5 + float(np.max(p[:6]))
        scores.append(area_conf * expr_weight)

    best_idx = int(np.argmax(scores))
    best_gray_112 = track[best_idx][4]

    # Trung bình có trọng số ưu tiên khung hình đỉnh biểu cảm (peak expression frame)
    weights = np.asarray(scores, dtype=np.float32)
    weights = weights / (weights.sum() + 1e-8)
    probs_accum = np.zeros(7, dtype=np.float32)
    for w, p in zip(weights, frame_probs):
        probs_accum += w * p
    probs_accum = 0.55 * frame_probs[best_idx] + 0.45 * probs_accum
    return best_gray_112, probs_accum / (probs_accum.sum() + 1e-8)


def _extract_full_audio_16k(video_path: Path, out_wav: Path) -> np.ndarray:
    """Tách kênh âm thanh 16kHz mono từ file video bằng ffmpeg (tự tìm binary)."""
    if not out_wav.exists():
        ffmpeg_bin = get_ffmpeg_binary()
        cmd = [
            ffmpeg_bin, "-y", "-i", str(video_path),
            "-ac", "1", "-ar", "16000", "-vn", str(out_wav)
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    y, _ = sf.read(str(out_wav), dtype="float32")
    if y.ndim > 1:
        y = y.mean(axis=1)
    return y


def is_clear_human_speech(wav: np.ndarray, sr: int = 16000, min_rms: float = 0.018) -> tuple[bool, float]:
    """Kiểm tra đoạn âm thanh 4s có thực sự chứa tiếng người nói rõ ràng (loại bỏ nhiễu/im lặng)."""
    rms = float(np.sqrt(np.mean(wav ** 2)))
    if rms < min_rms:
        return False, rms

    frame_len = int(0.05 * sr)
    n_frames = len(wav) // frame_len
    if n_frames < 10:
        return False, rms
    frames = wav[: n_frames * frame_len].reshape(n_frames, frame_len)
    frame_rms = np.sqrt(np.mean(frames ** 2, axis=1))

    p90 = float(np.percentile(frame_rms, 90))
    p15 = float(np.percentile(frame_rms, 15))
    active_ratio = float(np.mean(frame_rms > (min_rms * 0.75)))
    if p90 < min_rms * 1.25 or active_ratio < 0.22 or (p90 / (p15 + 1e-4)) < 1.35:
        return False, rms

    spec = np.abs(np.fft.rfft(wav)) ** 2
    freqs = np.fft.rfftfreq(len(wav), d=1.0 / sr)
    total_power = float(np.sum(spec)) + 1e-8
    speech_band = (freqs >= 100.0) & (freqs <= 4000.0)
    speech_ratio = float(np.sum(spec[speech_band])) / total_power
    if speech_ratio < 0.45:
        return False, rms

    return True, rms


def clean_and_denoise_audio(
    wav: np.ndarray,
    sr: int = 16000,
    low_cut: float = 90.0,
    high_cut: float = 7200.0,
    denoise: bool = True,
) -> np.ndarray:
    """Lọc dải thông giọng nói (90Hz - 7200Hz), giảm nhiễu nền tĩnh nhẹ nhàng."""
    try:
        from scipy import signal
        sos = signal.butter(4, [low_cut, high_cut], btype="bandpass", fs=sr, output="sos")
        filtered = signal.sosfiltfilt(sos, wav).astype(np.float32)
    except Exception:
        filtered = wav.astype(np.float32)

    if denoise:
        try:
            import noisereduce as nr
            filtered = nr.reduce_noise(y=filtered, sr=sr, stationary=True, prop_decrease=0.45)
        except Exception:
            pass

    max_val = float(np.max(np.abs(filtered)))
    if max_val > 1e-4:
        gain = min(3.0, 0.85 / max_val)
        filtered = np.clip(filtered * gain, -0.98, 0.98)
    return filtered.astype(np.float32)


def _find_subtitle_in_window(
    subtitles: list[tuple[float, float, str]], start_s: float, end_s: float
) -> tuple[str, bool]:
    """Lấy lời thoại tiếng Việt giao với khoảng [start_s, end_s], khử ký tự HTML (&gt;&gt;) và khử lặp câu cuốn chiếu."""
    import html
    import re

    raw_matched = []
    for s_start, s_end, text in subtitles:
        if s_end >= start_s and s_start <= end_s and text.strip():
            t = html.unescape(text).replace(">>", " ").replace("<<", " ").strip()
            t = re.sub(r"\s+", " ", t).strip()
            if t:
                raw_matched.append(t)
    if not raw_matched:
        return "", False

    dedup: list[str] = []
    for phrase in raw_matched:
        low_p = phrase.lower()
        if low_p in ("[âm nhạc]", "[vỗ tay]", "[cười]"):
            continue
        if dedup and (phrase in dedup[-1] or dedup[-1] in phrase):
            if len(phrase) > len(dedup[-1]):
                dedup[-1] = phrase
        else:
            dedup.append(phrase)

    cleaned = " ".join(dedup).strip()
    cleaned = re.sub(r"\[(?:âm nhạc|vỗ tay|cười)\]", " ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    # Khử hiện tượng lặp cụm từ cuốn chiếu (3..10 từ lặp liên tiếp) của phụ đề tự động YouTube
    for _ in range(2):
        cleaned = re.sub(r"\b(\S+(?:\s+\S+){2,9})\s+\1\b", r"\1", cleaned, flags=re.IGNORECASE)
    is_pure_music = (len(raw_matched) > 0 and not cleaned)
    return cleaned, is_pure_music


def _is_frontal_clear_face(f: np.ndarray, aligned_gray: np.ndarray) -> bool:
    """Kiểm tra khuôn mặt nhìn thẳng (frontal), không chống tay lên cằm nghe, không bị che vùng miệng."""
    w, h = float(f[2]), float(f[3])
    rx, ry = float(f[4]), float(f[5])
    lx, ly = float(f[6]), float(f[7])
    nx, ny = float(f[8]), float(f[9])
    rmx, rmy = float(f[10]), float(f[11])
    lmx, lmy = float(f[12]), float(f[13])

    eye_dx = abs(lx - rx)
    eye_dy = abs(ly - ry)
    mouth_dx = abs(lmx - rmx)

    if eye_dx < 0.24 * w:
        return False
    if eye_dy > 0.30 * eye_dx:
        return False
    min_ex, max_ex = min(rx, lx), max(rx, lx)
    if not (min_ex + 0.12 * eye_dx <= nx <= max_ex - 0.12 * eye_dx):
        return False
    if mouth_dx < 0.18 * w or min(rmy, lmy) <= ny:
        return False

    # Kiểm tra đối xứng hai bên má/cằm trên ảnh 112x112:
    # Cảnh chống tay lên cằm nghe có ĐỒNG THỜI đầu nghiêng (eye_dy > 0.135 * eye_dx) VÀ bất đối xứng má/cằm (cheek_asym > 28)
    left_cheek = aligned_gray[65:105, 18:46].astype(np.float32)
    right_cheek = np.fliplr(aligned_gray[65:105, 66:94]).astype(np.float32)
    cheek_asym = float(np.mean(np.abs(left_cheek - right_cheek)))
    if (eye_dy > 0.135 * eye_dx and cheek_asym > 28.0) or cheek_asym > 56.0:
        return False

    lower_half = aligned_gray[56:104, 24:88]
    if lower_half.size == 0:
        return False
    lower_lap = float(cv2.Laplacian(lower_half, cv2.CV_64F).var())
    lower_mean = float(lower_half.mean())
    if lower_lap < 25.0 or lower_mean > 218.0:
        return False
    return True


def _compute_mouth_speaking_activity(
    aligned_grays: list[np.ndarray],
    audio_sub_rms: np.ndarray,
) -> tuple[bool, float]:
    """Đo cử động đóng/mở miệng thực tế của CÙNG MỘT NGƯỜI trên chuỗi mặt 112x112 (so với vùng trán/mắt cố định).

    - Không chia chuẩn hóa cho std cục bộ của miệng (tránh khuếch đại nhiễu khi mím môi).
    - So sánh trực tiếp độ biến thiên vùng miệng `m_diff` (y∈[82..104], x∈[38..74]) với vùng mắt/mũi `up_diff` (y∈[18..70], x∈[22..90]).
    - Đòi hỏi độ mở hốc miệng (`mb.std()`) phải dao động đóng-mở thực sự giữa các âm tiết (`std_osc >= 2.2`).
    """
    if len(aligned_grays) < 3:
        return False, 0.0

    up_patches = [g[18:70, 22:90].astype(np.float32) for g in aligned_grays]
    mouth_patches = [g[82:104, 38:74].astype(np.float32) for g in aligned_grays]
    inner_stds = [float(g[84:100, 42:70].astype(np.float32).std()) for g in aligned_grays]

    up_diffs = [float(np.mean(np.abs(up_patches[i + 1] - up_patches[i]))) for i in range(len(up_patches) - 1)]
    m_diffs = [float(np.mean(np.abs(mouth_patches[i + 1] - mouth_patches[i]))) for i in range(len(mouth_patches) - 1)]

    mean_up_diff = float(np.mean(up_diffs))
    mean_m_diff = float(np.mean(m_diffs))
    max_m_diff = float(np.max(m_diffs))

    # Tỷ lệ cử động miệng so với cử động cứng của đầu/mắt
    mouth_to_upper_ratio = mean_m_diff / (mean_up_diff + 1.5)

    # Biên độ dao động đóng - mở hốc miệng qua các khung hình (mím môi tĩnh sẽ có std_osc < 1.5)
    std_osc = float(np.max(inner_stds) - np.min(inner_stds))

    # Tương quan đồng bộ Miệng - Âm thanh
    open_arr = np.asarray(inner_stds, dtype=np.float32)
    corr = 0.0
    if len(audio_sub_rms) == len(open_arr) and open_arr.std() > 0.5 and audio_sub_rms.std() > 1e-5:
        c_mat = np.corrcoef(open_arr, audio_sub_rms)
        if not np.isnan(c_mat[0, 1]):
            corr = float(c_mat[0, 1])

    lip_score = (mean_m_diff / 12.0) * (0.7 + 0.3 * mouth_to_upper_ratio) * (1.0 + 0.25 * max(-0.2, corr))

    is_speaking = (
        mean_m_diff >= 7.2
        and max_m_diff >= 11.0
        and mouth_to_upper_ratio >= 1.04
        and std_osc >= 1.8
        and lip_score >= 0.55
    )
    return is_speaking, round(lip_score, 4)


def extract_face_audio_clips(
    video: DownloadedVideo,
    out_dir: str | Path = "data/processed/vn_youtube",
    clip_duration: float = 4.0,
    step_sec: float = 4.0,
    min_rms: float = 0.018,
    min_face_size: int = 48,
    max_clips_per_video: int = 20,
) -> list[ExtractedClip]:
    """Duyệt qua video theo cửa sổ 4s, chỉ giữ đoạn mà người trong khung hình THỰC SỰ ĐANG MỞ MIỆNG NÓI khớp tiếng."""
    out_root = Path(out_dir)
    face_dir = out_root / "faces"
    audio_dir = out_root / "audio"
    face_dir.mkdir(parents=True, exist_ok=True)
    audio_dir.mkdir(parents=True, exist_ok=True)

    full_wav_path = video.video_path.with_suffix(".16k.wav")
    try:
        audio_16k = _extract_full_audio_16k(video.video_path, full_wav_path)
    except Exception as e:
        print(f"  [!] Không tách được âm thanh từ {video.video_id}: {e}")
        return []

    sr = 16000
    total_sec = len(audio_16k) / sr

    cap = cv2.VideoCapture(str(video.video_path))
    if not cap.isOpened():
        return []
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 640)
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 360)

    yunet_path = ensure_yunet_model()
    detector = cv2.FaceDetectorYN.create(str(yunet_path), "", (frame_w, frame_h), 0.80, 0.3, 5000)

    fer_net = None
    try:
        fer_path = ensure_fer_model()
        fer_net = cv2.dnn.readNet(str(fer_path))
    except Exception as e:
        print(f"  [!] Không nạp được FER ONNX ({e}), dùng geometric fallback.")

    clahe = cv2.createCLAHE(clipLimit=1.8, tileGridSize=(4, 4))

    clips: list[ExtractedClip] = []
    clip_idx = 0
    t_start = 2.0

    # Lấy 7 mốc thời gian rải đều trong đoạn 4 giây để theo dõi cử động miệng liên tục
    rel_points = (0.12, 0.25, 0.38, 0.50, 0.62, 0.75, 0.88)
    cur_frame_pos = 0

    while t_start + clip_duration <= total_sec and len(clips) < max_clips_per_video:
        t_end = t_start + clip_duration
        s_idx = int(t_start * sr)
        e_idx = int(t_end * sr)
        wav_chunk = audio_16k[s_idx:e_idx]

        # 1. Kiểm tra phụ đề (bỏ qua nếu chỉ có [âm nhạc]) và tiếng người nói thật
        transcript, is_pure_music = _find_subtitle_in_window(video.subtitles, t_start, t_end)
        if is_pure_music:
            t_start += step_sec
            continue

        ok_speech, rms = is_clear_human_speech(wav_chunk, sr=sr, min_rms=min_rms)
        if not ok_speech:
            t_start += step_sec
            continue

        # 2. Theo dõi các khuôn mặt (Face Tracks) qua 7 khung hình trong đoạn 4 giây
        # Mỗi track lưu: list các tuple (t_idx, center_x, center_y, aligned_bgr_112, aligned_gray_112, upper_gray_112, sharpness, area_conf)
        tracks: list[list[tuple[int, float, float, np.ndarray, np.ndarray, np.ndarray, float, float]]] = []
        sub_rms_list: list[float] = []

        for r_idx, r in enumerate(rel_points):
            st = t_start + clip_duration * r
            # Năng lượng âm thanh trong cửa sổ 0.35s quanh thời điểm khung hình này
            w0 = int(max(0, (clip_duration * r - 0.175) * sr))
            w1 = int(min(len(wav_chunk), (clip_duration * r + 0.175) * sr))
            sub_rms = float(np.sqrt(np.mean(wav_chunk[w0:w1] ** 2))) if w1 > w0 else rms
            sub_rms_list.append(sub_rms)

            target_frame = int(st * fps)
            diff_frames = target_frame - cur_frame_pos
            if 0 <= diff_frames <= int(fps * 8):
                # Tiến tuần tự bằng grab() để tránh lỗi giải mã H.264 seek và nhanh gấp 10 lần
                ok = True
                for _ in range(diff_frames):
                    if not cap.grab():
                        ok = False
                        break
                if ok:
                    ok, frame = cap.retrieve()
                    cur_frame_pos = target_frame + 1
                else:
                    frame = None
            else:
                cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
                ok, frame = cap.read()
                cur_frame_pos = target_frame + 1

            if not ok or frame is None:
                continue

            h_f, w_f = frame.shape[:2]
            if (w_f, h_f) != (frame_w, frame_h):
                frame_w, frame_h = w_f, h_f
                detector.setInputSize((frame_w, frame_h))

            _, faces = detector.detect(frame)
            if faces is None or len(faces) == 0:
                continue

            gray_full = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            for f in faces:
                conf = float(f[-1])
                w, h = float(f[2]), float(f[3])
                if conf < 0.84 or w < min_face_size or h < min_face_size:
                    continue

                # Căn chỉnh khuôn mặt chuẩn 112x112 theo 5 điểm mốc YuNet
                lm5 = f[4:14].reshape(5, 2)
                try:
                    aligned_bgr = align_face_112(frame, lm5)
                except Exception:
                    continue
                aligned_gray = cv2.cvtColor(aligned_bgr, cv2.COLOR_BGR2GRAY)

                if not _is_frontal_clear_face(f, aligned_gray):
                    continue
                sharpness = float(cv2.Laplacian(aligned_gray, cv2.CV_64F).var())
                if float(aligned_gray.std()) < 22.0 or sharpness < 22.0:
                    continue

                cx = float(f[0]) + w * 0.5
                cy = float(f[1]) + h * 0.5
                area_conf = (w * h) * sharpness * conf

                # Vector đặc trưng vùng mắt-mũi-trán chuẩn hóa để chống gộp nhầm 2 người khi chuyển góc máy (Camera Cut)
                up_patch = aligned_gray[18:70, 22:90].astype(np.float32)
                up_norm = (up_patch - float(up_patch.mean())) / (float(up_patch.std()) + 1e-4)

                matched_track = None
                for tr in tracks:
                    last_r_idx, last_cx, last_cy, _, _, last_up_norm, _, _ = tr[-1]
                    if last_r_idx == r_idx:
                        continue
                    if abs(cx - last_cx) < 0.22 * frame_w and abs(cy - last_cy) < 0.22 * frame_h:
                        id_corr = float(np.mean(up_norm * last_up_norm))
                        if id_corr >= 0.58:
                            matched_track = tr
                            break
                if matched_track is not None:
                    matched_track.append((r_idx, cx, cy, aligned_bgr, aligned_gray, up_norm, sharpness, area_conf))
                else:
                    tracks.append([(r_idx, cx, cy, aligned_bgr, aligned_gray, up_norm, sharpness, area_conf)])

        # 3. Đánh giá từng Face Track (của cùng 1 người) xem ai là người THỰC SỰ ĐANG MỞ MIỆNG NÓI (Active Speaker)
        best_track = None
        best_lip_score = -1.0

        for tr in tracks:
            if len(tr) < 3:
                continue
            tr_aligned_grays = [item[4] for item in tr]
            tr_rms = np.asarray([sub_rms_list[item[0]] for item in tr], dtype=np.float32)
            is_speaking, lip_score = _compute_mouth_speaking_activity(tr_aligned_grays, tr_rms)
            if is_speaking and lip_score > best_lip_score:
                best_lip_score = lip_score
                best_track = tr

        if best_track is None:
            t_start += step_sec
            continue

        # 4. Chọn khung hình đỉnh biểu cảm (Peak Expression Frame) của người đang nói & nhận diện sắc thái mặt
        best_gray_112, face_expr_probs = select_peak_expressive_frame_and_probs(
            fer_net, best_track, video.category_emotion
        )
        best_face_48 = cv2.resize(clahe.apply(best_gray_112), (48, 48), interpolation=cv2.INTER_AREA)

        sample_id = f"{video.video_id}_{clip_idx:04d}"
        face_file = face_dir / f"{sample_id}.png"
        audio_file = audio_dir / f"{sample_id}.wav"

        clean_wav = clean_and_denoise_audio(wav_chunk, sr=sr)
        Image.fromarray(best_face_48).save(face_file)
        sf.write(str(audio_file), clean_wav, sr)

        clips.append(
            ExtractedClip(
                sample_id=sample_id,
                video_id=video.video_id,
                speaker=f"vid_{video.video_id}",
                start_sec=round(t_start, 2),
                end_sec=round(t_end, 2),
                face_path=str(face_file.as_posix()),
                audio_path=str(audio_file.as_posix()),
                transcript_vi=transcript,
                rms_energy=round(rms, 4),
                category_emotion=video.category_emotion,
                lip_sync_score=best_lip_score,
                face_expr_probs=[round(float(x), 4) for x in face_expr_probs.tolist()],
            )
        )
        clip_idx += 1
        # Sau khi lấy được 1 đoạn 4s hợp lệ, nhảy ít nhất (clip_duration + 1.0s) để không bao giờ bị trùng lặp câu thoại!
        t_start += max(step_sec, clip_duration + 1.0)

    cap.release()
    return clips
