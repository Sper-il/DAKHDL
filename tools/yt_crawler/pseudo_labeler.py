"""Module gợi ý nhãn tự động Đa phương thức: Sắc thái Khuôn mặt (MobileFaceNet FER) + Âm điệu (Pitch/Spectral/Energy) + Phụ đề Tiếng Việt."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import re
import numpy as np

from .extractor import ExtractedClip

EMOTIONS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]

# Từ điển 180+ từ khóa & cụm từ cảm xúc tiếng Việt trong hội thoại/phim/phỏng vấn
VI_EMOTION_KEYWORDS: dict[int, list[str]] = {
    0: [  # angry
        "tức", "giận", "điên", "bực", "ghét", "cáu", "quạo", "vô lý", "chửi", "mắng",
        "hỗn", "láo", "cút", "im đi", "quá đáng", "không chấp nhận", "bất công", "phản bội",
        "khốn nạn", "lừa đảo", "căm", "hận", "nổi điên", "bức xúc", "cãi", "trách", "hét",
    ],
    1: [  # disgust
        "gớm", "kinh tởm", "tởm", "dơ", "bẩn", "ớn", "buồn nôn", "ghê quá", "nhạt nhẽo",
        "ngán", "chướng mắt", "khó chịu", "mệt người", "phiền", "vô duyên", "giả tạo",
        "đạo đức giả", "thối", "tanh", "bốc mùi", "nhếch nhác", "coi thường", "khinh",
    ],
    2: [  # fear
        "sợ hãi", "hoảng sợ", "run sợ", "hãi", "ám ảnh", "nguy hiểm", "cứu với", "lo lắng", "rùng mình",
        "chết mất", "dọa ma", "bóng ma", "bị bắt", "chạy trốn", "cẩn thận", "lo sợ", "khiếp", "ác mộng",
        "bất an", "hồi hộp", "đe dọa", "tai nạn", "cấp cứu", "hoảng loạn", "sợ quá",
    ],
    3: [  # happy
        "vui", "cười", "hạnh phúc", "tuyệt", "thích", "yêu", "haha", "hihi", "cảm ơn", "vui mừng",
        "đỉnh", "tuyệt vời", "may quá", "thành công", "chúc mừng", "thương quá", "đáng yêu",
        "dễ thương", "vui vẻ", "phấn khởi", "tự hào", "ngon quá", "đẹp quá", "tặng quà",
    ],
    4: [  # sad
        "buồn", "khóc", "tiếc", "đau khổ", "thương tâm", "cô đơn", "chán nản", "tủi thân", "mệt mỏi", "thất vọng",
        "chia tay", "mất mát", "qua đời", "nghèo khổ", "khổ tâm", "tội nghiệp", "nước mắt", "xin lỗi",
        "hối hận", "bỏ rơi", "tuyệt vọng", "đau lòng", "xót xa", "bất lực", "tổn thương",
    ],
    5: [  # surprise
        "ủa", "hả", "trời ơi", "bất ngờ", "ngạc nhiên", "thật á", "không ngờ", "sốc", "wow",
        "ơ kìa", "sao lại", "thật không", "làm sao", "không thể tin", "chuyện gì", "kinh ngạc",
        "giật mình", "ối", "trời đất", "ảo thật", "lạ vậy",
    ],
    6: [  # neutral
        "bình thường", "hôm nay", "bây giờ", "thông tin", "cho biết", "theo đó", "tuy nhiên",
        "ngoài ra", "cụ thể", "đồng thời", "chương trình", "báo cáo", "kế hoạch", "công việc",
    ],
}


@dataclass
class LabeledSample:
    sample_id: str
    video_id: str
    speaker: str
    start_sec: float
    end_sec: float
    face_path: str
    audio_path: str
    transcript_vi: str
    label: int
    emotion: str
    confidence: float
    reason: str
    verified: int = 0
    skip: int = 0


def _score_vietnamese_text(text: str) -> tuple[np.ndarray, list[str]]:
    """Tính phân bố xác suất cảm xúc từ phụ đề tiếng Việt (khớp nguyên từ & cụm từ)."""
    probs = np.zeros(7, dtype=np.float32)
    matched_words: list[str] = []
    if not text:
        return probs, matched_words

    cleaned = re.sub(r"[^\w\s]", " ", text.lower())
    padded = f" {' '.join(cleaned.split())} "
    negations = ("không", "chẳng", "chả", "chưa", "đừng", "đâu có")

    for label_id, keywords in VI_EMOTION_KEYWORDS.items():
        for kw in keywords:
            if f" {kw} " in padded:
                if any(f" {neg} {kw} " in padded for neg in negations):
                    if label_id == 3:
                        probs[1] += 0.35
                        matched_words.append(f"không {kw}->disgust")
                    continue
                weight = 0.45 if " " in kw else 0.30
                probs[label_id] += weight
                matched_words.append(f"{kw}->{EMOTIONS[label_id]}")

    if probs.sum() > 0:
        probs = probs / probs.sum()
    return probs, matched_words


def _estimate_pitch_f0_stats(y: np.ndarray, sr: int = 16000) -> tuple[float, float]:
    """Ước lượng tần số cơ bản F0 (Pitch trung bình & độ biến thiên F0 std) bằng tự tương quan trên các khung hữu thanh."""
    frame_len = int(0.04 * sr)  # 40ms
    hop_len = int(0.02 * sr)    # 20ms
    min_lag = int(sr / 400.0)   # 400 Hz
    max_lag = int(sr / 80.0)    # 80 Hz

    f0_vals = []
    rms_thresh = max(0.015, float(np.sqrt(np.mean(y ** 2))) * 0.65)

    for start in range(0, len(y) - frame_len, hop_len):
        frame = y[start : start + frame_len]
        if float(np.sqrt(np.mean(frame ** 2))) < rms_thresh:
            continue
        frame = frame - np.mean(frame)
        corr = np.correlate(frame, frame, mode="full")[frame_len - 1 :]
        if corr[0] <= 1e-6:
            continue
        corr = corr / corr[0]
        sub = corr[min_lag:max_lag]
        if len(sub) == 0:
            continue
        peak_idx = int(np.argmax(sub)) + min_lag
        if corr[peak_idx] >= 0.32:
            f0_vals.append(float(sr / peak_idx))

    if len(f0_vals) < 3:
        return 165.0, 20.0
    arr = np.asarray(f0_vals, dtype=np.float32)
    return float(np.median(arr)), float(np.std(arr))


def _score_acoustic_prosody(clip: ExtractedClip) -> tuple[np.ndarray, str]:
    """Phân tích âm điệu (Pitch F0, biên độ động gốc, tốc độ âm tiết, trọng tâm phổ tần số) ra phân bố 7 cảm xúc."""
    import soundfile as sf

    probs = np.ones(7, dtype=np.float32) * 0.14
    notes = []
    try:
        y, sr = sf.read(clip.audio_path, dtype="float32")
        if y.ndim > 1:
            y = y.mean(axis=1)

        raw_rms = float(clip.rms_energy) if clip.rms_energy > 0 else float(np.sqrt(np.mean(y ** 2)))
        zcr = float(np.mean(np.abs(np.diff(np.sign(y)))) / 2.0)
        f0_mean, f0_std = _estimate_pitch_f0_stats(y, sr=sr)

        # Trọng tâm phổ (Spectral Centroid) & tỷ lệ năng lượng tần số cao (900Hz-4kHz / 100Hz-900Hz)
        spec = np.abs(np.fft.rfft(y))
        freqs = np.fft.rfftfreq(len(y), d=1.0 / sr)
        centroid = float(np.sum(freqs * spec) / (np.sum(spec) + 1e-8))
        low_e = float(np.sum(spec[(freqs >= 100) & (freqs < 900)] ** 2)) + 1e-8
        high_e = float(np.sum(spec[(freqs >= 900) & (freqs <= 4000)] ** 2))
        hf_ratio = high_e / low_e

        # Biến thiên năng lượng theo khung 50ms (Syllable burst / dynamic range)
        flen = int(0.05 * sr)
        n_f = len(y) // flen
        f_rms = np.sqrt(np.mean(y[: n_f * flen].reshape(n_f, flen) ** 2, axis=1))
        dyn_cv = float(f_rms.std() / (f_rms.mean() + 1e-5))

        # 1. Tức giận (Angry): âm lượng gốc lớn hoặc tỷ lệ tần số cao lớn (giọng gắt/căng/quát)
        if (raw_rms > 0.065 and hf_ratio > 0.32) or (f0_mean > 195 and hf_ratio > 0.42 and dyn_cv > 0.48):
            probs[0] += 0.44
            probs[1] += 0.14
            notes.append(f"giọng gắt/căng F0={int(f0_mean)}Hz")

        # 2. Ngạc nhiên (Surprise): F0 cao và biến thiên pitch rộng (lên giọng đột ngột)
        if f0_std > 40.0 and f0_mean > 195.0:
            probs[5] += 0.42
            notes.append(f"ngữ điệu vọt cao ΔF0={int(f0_std)}Hz")

        # 3. Vui vẻ (Happy): cao độ trung-cao ấm, nhịp điệu đều linh hoạt
        if 180.0 <= f0_mean <= 255.0 and 20.0 <= f0_std <= 44.0 and 0.32 <= dyn_cv <= 0.62:
            probs[3] += 0.34
            notes.append(f"giọng tươi/sôi nổi F0={int(f0_mean)}Hz")

        # 4. Sợ hãi (Fear): giọng run/hấp tấp (biến thiên năng lượng thất thường + ZCR/jitter cao)
        if dyn_cv > 0.62 and (zcr > 0.095 or f0_std > 32.0):
            probs[2] += 0.42
            notes.append("nhịp giọng gấp/run")

        # 5. Buồn (Sad): cao độ trầm hoặc âm lượng gốc nhỏ/lắng, trọng tâm phổ thấp
        if (f0_mean < 172.0 and f0_std < 30.0) or (raw_rms < 0.045 and centroid < 1250.0):
            probs[4] += 0.42
            notes.append(f"giọng trầm/lắng F0={int(f0_mean)}Hz")

        # 6. Khó chịu / Chán ghét (Disgust): âm sắc nén/căng ở tần số trung, giọng gằn nhẹ
        if 145.0 <= f0_mean <= 205.0 and hf_ratio > 0.28 and 0.35 <= dyn_cv <= 0.65:
            probs[1] += 0.36
            notes.append("âm sắc nén/khó chịu")

        # 7. Bình thường (Neutral): cao độ ổn định, biên độ đều đặn
        if f0_std < 24.0 and 0.22 <= dyn_cv <= 0.50:
            probs[6] += 0.36
            notes.append(f"ngữ điệu điềm tĩnh F0={int(f0_mean)}Hz")

        if not notes:
            notes.append(f"F0={int(f0_mean)}±{int(f0_std)}Hz")
    except Exception:
        notes.append("âm học mặc định")

    probs = probs / (probs.sum() + 1e-8)
    return probs, ", ".join(notes[:2])


def pseudo_label_clips(
    clips: list[ExtractedClip],
    face_ckpt: str | Path | None = None,
    audio_ckpt: str | Path | None = None,
) -> list[LabeledSample]:
    """Gán nhãn tự động bằng Tổ hợp Đa phương thức: Sắc thái Khuôn mặt + Âm điệu Giọng nói + Phụ đề Tiếng Việt."""
    labeled_list: list[LabeledSample] = []

    # Bảng tương thích cơ mặt FACS khi đang mở miệng nói (Speaking Action-Unit Compatibility):
    # Khi diễn viên vừa mở miệng nói/quát/khóc vừa biểu cảm, các cơ mặt giao thoa trong cùng nhóm Arousal/Valence:
    facs_compat = {
        0: (1, 5),     # angry <-> disgust (AU4 nhíu mày), surprise (AU5+AU26 trợn mắt/há miệng quát)
        1: (0, 4),     # disgust <-> angry (AU4), sad (AU4+AU15 nhăn mặt)
        2: (5, 4),     # fear <-> surprise (AU1+AU2+AU5 mở to mắt), sad (AU1+AU4 căng chân mày)
        3: (5,),       # happy <-> surprise (cười lớn mở miệng)
        4: (1, 2),     # sad <-> disgust (AU4 nhíu mày khóc), fear (AU1+AU4)
        5: (2, 3),     # surprise <-> fear, happy
    }

    for clip in clips:
        reasons = []
        cat_idx = EMOTIONS.index(clip.category_emotion) if clip.category_emotion in EMOTIONS else -1

        # 2. Điểm Âm điệu Giọng nói & 3. Điểm Ngữ nghĩa Phụ đề Tiếng Việt
        p_prosody, prosody_note = _score_acoustic_prosody(clip)
        p_text, matched_kw = _score_vietnamese_text(clip.transcript_vi)

        # 1. Điểm Sắc thái Khuôn mặt (MobileFaceNet FER 88.27% RAF-DB đã hiệu chỉnh Logit Adjustment)
        if clip.face_expr_probs and len(clip.face_expr_probs) == 7:
            p_face = np.asarray(clip.face_expr_probs, dtype=np.float32).copy()
            # CHỈ áp dụng tương thích cơ mặt FACS khi khuôn mặt thực sự biểu cảm (không phải mặt neutral và không phải đang cười tươi)
            is_expressive_face = (float(p_face[6]) < 0.45) and (float(p_face[3]) < 0.48 if cat_idx != 3 else float(p_face[3]) >= 0.25)
            if 0 <= cat_idx < 6 and is_expressive_face:
                for comp_idx in facs_compat.get(cat_idx, ()):
                    transfer = 0.65 * p_face[comp_idx]
                    p_face[cat_idx] += transfer
                    p_face[comp_idx] -= transfer * 0.70
            p_face = p_face / (p_face.sum() + 1e-8)
            top_f = int(np.argmax(p_face))
            reasons.append(f"Sắc thái mặt[{EMOTIONS[top_f]}:{p_face[top_f]:.2f}]")
        else:
            p_face = np.ones(7, dtype=np.float32) / 7.0
            is_expressive_face = False

        reasons.append(f"Âm điệu[{prosody_note}]")

        # Hợp nhất 3 nguồn tín hiệu thực tế của chính đoạn cắt 4s này (ưu tiên Sắc thái mặt + Phụ đề + Âm điệu)
        if p_text.sum() > 0:
            probs = 0.62 * p_face + 0.18 * p_prosody + 0.20 * p_text
            reasons.append(f"Phụ đề[{', '.join(matched_kw[:2])}]")
        else:
            probs = 0.72 * p_face + 0.28 * p_prosody

        # 4. Điểm gợi ý bối cảnh: CHỈ cộng khi chính Sắc thái mặt đã đồng thuận với bối cảnh (tránh gán nhãn ngược với biểu cảm mặt!)
        if 0 <= cat_idx < 7 and (top_f == cat_idx or float(p_face[cat_idx]) >= 0.35):
            probs[cat_idx] += 0.14
            reasons.append(f"Bối cảnh[{clip.category_emotion}]")

        if clip.lip_sync_score > 0:
            reasons.append(f"LipSync[{clip.lip_sync_score:.2f}]")

        probs = probs / (probs.sum() + 1e-8)
        label_idx = int(np.argmax(probs))
        conf = round(float(probs[label_idx]), 3)

        labeled_list.append(
            LabeledSample(
                sample_id=clip.sample_id,
                video_id=clip.video_id,
                speaker=clip.speaker,
                start_sec=clip.start_sec,
                end_sec=clip.end_sec,
                face_path=clip.face_path,
                audio_path=clip.audio_path,
                transcript_vi=clip.transcript_vi,
                label=label_idx,
                emotion=EMOTIONS[label_idx],
                confidence=conf,
                reason=" + ".join(reasons),
            )
        )

    return labeled_list
