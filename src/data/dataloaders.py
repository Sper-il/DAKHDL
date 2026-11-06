"""DataLoader cho 2 nhánh mặt và tiếng."""
from torch.utils.data import DataLoader
from .face_dataset import FaceDataset
from .audio_dataset import AudioDataset


def build_face_loader(
    csv_path: str,
    batch_size: int = 64,
    shuffle: bool = True,
    num_workers: int = 4,
    transform=None,
) -> DataLoader:
    ds = FaceDataset(csv_path, transform=transform)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )


def build_audio_loader(
    csv_path: str,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 4,
    transform=None,
    augment_wave: bool = False,
) -> DataLoader:
    ds = AudioDataset(csv_path, transform=transform, augment_wave=augment_wave)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )
