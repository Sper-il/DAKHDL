"""Thứ tự nhãn dùng chung cho cả nhánh mặt và nhánh tiếng (chỉ số 0..9)."""

EMOTIONS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral",
            "contempt", "boredom", "excited"]

EMOTIONS_VI = ["Giận dữ", "Ghê tởm", "Sợ hãi", "Vui vẻ", "Buồn", "Ngạc nhiên",
               "Trung tính", "Khinh bỉ", "Chán nản", "Hào hứng"]

# Bộ 4 lớp cho DataViet (tư vấn/tâm sự tiếng Việt) & ánh xạ từ mô hình mặt 7 lớp
EMOTIONS_4 = ["angry", "happy", "sad", "neutral"]
EMOTIONS_4_VI = ["Giận dữ", "Vui vẻ", "Buồn", "Trung tính"]
FACE7_TO_4_IDX = [0, 3, 4, 6]


def class_names(num_classes: int):
    """Tên của `num_classes` lớp (4 = bộ 4 cảm xúc tâm sự, 7 = cơ bản, 10 = đủ bộ)."""
    if num_classes == 4:
        return EMOTIONS_4
    if not 2 <= num_classes <= len(EMOTIONS):
        raise ValueError(f"num_classes phải trong khoảng 2..{len(EMOTIONS)}, nhận {num_classes}")
    return EMOTIONS[:num_classes]


def class_names_vi(num_classes: int):
    """Tên tiếng Việt tương ứng với `class_names(num_classes)`."""
    if num_classes == 4:
        return EMOTIONS_4_VI
    return EMOTIONS_VI[:num_classes]
