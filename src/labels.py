"""Thứ tự nhãn dùng chung cho cả nhánh mặt và nhánh tiếng (chỉ số 0..9)."""

EMOTIONS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral",
            "contempt", "boredom", "excited"]

EMOTIONS_VI = ["Giận dữ", "Ghê tởm", "Sợ hãi", "Vui vẻ", "Buồn", "Ngạc nhiên",
               "Trung tính", "Khinh bỉ", "Chán nản", "Hào hứng"]


def class_names(num_classes: int):
    """Tên của `num_classes` lớp đầu tiên (7 = 7 cảm xúc cơ bản, 10 = đủ bộ)."""
    if not 2 <= num_classes <= len(EMOTIONS):
        raise ValueError(f"num_classes phải trong khoảng 2..{len(EMOTIONS)}, nhận {num_classes}")
    return EMOTIONS[:num_classes]
