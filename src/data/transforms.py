"""Augment cho 2 nhánh: mặt dùng Albumentations, tiếng dùng SpecAugment nhẹ."""
import torch


def build_train_face_transform():
    """Augment train nhánh mặt: flip, xoay nhẹ, sáng/tương phản."""
    from albumentations import Compose, HorizontalFlip, ShiftScaleRotate, RandomBrightnessContrast
    from albumentations.pytorch import ToTensorV2
    import numpy as np
    from PIL import Image

    aug = Compose([
        HorizontalFlip(p=0.5),
        ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=10, p=0.5),
        RandomBrightnessContrast(p=0.3),
    ])

    def _fn(pil_img: Image.Image) -> torch.Tensor:
        arr = np.array(pil_img.convert("L").resize((48, 48)))
        out = aug(image=arr)["image"]
        t = torch.from_numpy(out).float().unsqueeze(0) / 255.0
        return (t - 0.5) / 0.5
    return _fn


def build_val_face_transform():
    """Val/test mặt: chỉ resize + chuẩn hóa, không augment."""
    from torchvision import transforms as T
    return T.Compose([T.Resize((48, 48)), T.ToTensor(), T.Normalize([0.5], [0.5])])


def spec_augment(mel: torch.Tensor, time_mask: int = 20, freq_mask: int = 16) -> torch.Tensor:
    """Mask ngẫu nhiên trên mel [1,F,T], chỉ dùng cho train."""
    import random
    out = mel.clone()
    _, f, t = out.shape
    tm = random.randint(0, time_mask)
    t0 = random.randint(0, max(0, t - tm))
    out[:, :, t0:t0 + tm] = out.mean()
    fm = random.randint(0, freq_mask)
    f0 = random.randint(0, max(0, f - fm))
    out[:, f0:f0 + fm, :] = out.mean()
    return out
