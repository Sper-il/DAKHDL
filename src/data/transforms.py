"""Augmentation cho ảnh mặt (48x48) và Mel-spectrogram (128x128) chống overfitting."""
from __future__ import annotations
import random
import numpy as np
from PIL import Image
import torch
from torchvision import transforms as T


def build_train_face_transform():
    """Augment train nhánh mặt: Flip, Shift/Scale/Rotate (±15°), Brightness/Contrast và RandomErasing (p=0.25)."""
    try:
        from albumentations import Compose, HorizontalFlip, ShiftScaleRotate, RandomBrightnessContrast

        aug = Compose([
            HorizontalFlip(p=0.5),
            ShiftScaleRotate(shift_limit=0.08, scale_limit=0.10, rotate_limit=15, border_mode=0, p=0.6),
            RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.4),
        ])
        erasing = T.RandomErasing(p=0.25, scale=(0.02, 0.14), ratio=(0.3, 3.3), value=0.0)

        def _fn(pil_img: Image.Image) -> torch.Tensor:
            arr = np.array(pil_img.convert("L").resize((48, 48)))
            out = aug(image=arr)["image"]
            t = torch.from_numpy(out).float().unsqueeze(0) / 255.0
            t = (t - 0.5) / 0.5
            return erasing(t)

        return _fn
    except ImportError:
        # Fallback thuần torchvision nếu môi trường chưa cài albumentations
        return T.Compose([
            T.Resize((48, 48)),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomAffine(degrees=15, translate=(0.08, 0.08), scale=(0.9, 1.1)),
            T.ColorJitter(brightness=0.2, contrast=0.2),
            T.ToTensor(),
            T.Normalize([0.5], [0.5]),
            T.RandomErasing(p=0.25, scale=(0.02, 0.14), ratio=(0.3, 3.3), value=0.0),
        ])


def build_val_face_transform():
    """Val/test mặt: chỉ resize + chuẩn hóa, không augment."""
    return T.Compose([T.Resize((48, 48)), T.ToTensor(), T.Normalize([0.5], [0.5])])


def spec_augment(mel: torch.Tensor, time_mask: int = 20, freq_mask: int = 16, num_masks: int = 2) -> torch.Tensor:
    """SpecAugment: che ngẫu nhiên dải tần số và thời gian trên Mel [1,F,T], chỉ dùng cho train."""
    out = mel.clone()
    _, f, t = out.shape
    fill_val = float(out.mean())
    for _ in range(num_masks):
        tm = random.randint(0, time_mask)
        if tm > 0:
            t0 = random.randint(0, max(0, t - tm))
            out[:, :, t0 : t0 + tm] = fill_val
        fm = random.randint(0, freq_mask)
        if fm > 0:
            f0 = random.randint(0, max(0, f - fm))
            out[:, f0 : f0 + fm, :] = fill_val
    return out


def build_train_audio_transform(time_mask: int = 20, freq_mask: int = 16, num_masks: int = 2):
    """Trả về hàm SpecAugment để gắn trực tiếp vào AudioDataset khi train."""
    def _fn(mel: torch.Tensor) -> torch.Tensor:
        return spec_augment(mel, time_mask=time_mask, freq_mask=freq_mask, num_masks=num_masks)
    return _fn
