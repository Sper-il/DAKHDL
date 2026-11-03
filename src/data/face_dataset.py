"""Dataset nhánh mặt FER2013: ảnh xám 48x48."""
from pathlib import Path
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset


EMOTIONS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]


class FaceDataset(Dataset):
    """Đọc ảnh mặt đã xử lý hoặc parse trực tiếp từ fer2013.csv.

    Output mỗi mẫu: ``{'image': tensor [1,48,48], 'label': int}``.
    """

    def __init__(self, csv_path: str | Path, transform=None):
        self.df = pd.read_csv(csv_path)
        if "pixels" in self.df.columns:
            # Định dạng FER2013 gốc: cột emotion,pixels[,Usage]
            self._mode = "pixels"
        else:
            # Định dạng sau xử lý: cột filepath,label
            self._mode = "files"
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        if self._mode == "pixels":
            pixels = [int(v) for v in str(row["pixels"]).split()]
            image = Image.new("L", (48, 48))
            image.putdata(pixels)
            label_key = "emotion" if "emotion" in row.index else "label"
            label = int(row[label_key])
        else:
            image = Image.open(row["filepath"]).convert("L")
            label = int(row["label"])
        if self.transform is not None:
            image = self.transform(image)
        else:
            # Mặc định: về tensor [1,48,48], chuẩn hóa mean 0.5 std 0.5
            import torchvision.transforms as T
            image = T.Compose([T.Resize((48, 48)), T.ToTensor(), T.Normalize([0.5], [0.5])])(image)
        return {"image": image, "label": torch.tensor(label, dtype=torch.long)}
