"""Dataset nhánh tiếng: WAV 16kHz mono -> Mel-spectrogram 128x128."""
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset

EMOTIONS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]


class AudioDataset(Dataset):
    """Đọc file npy mel đã xử lý hoặc WAV gốc rồi trích mel on-the-fly.

    Output mỗi mẫu: ``{'mel': tensor [1,128,128], 'label': int}``.
    """

    def __init__(self, csv_path: str | Path, transform=None, n_mels: int = 128, n_frames: int = 128):
        import pandas as pd
        self.df = pd.read_csv(csv_path)
        self.transform = transform
        self.n_mels = n_mels
        self.n_frames = n_frames

    def __len__(self) -> int:
        return len(self.df)

    def _load_mel(self, path: str) -> np.ndarray:
        p = str(path)
        if p.endswith(".npy"):
            mel = np.load(p).astype(np.float32)
        else:
            # WAV gốc: trích mel bằng librosa
            import librosa
            y, sr = librosa.load(p, sr=16000, mono=True, duration=4.0)
            target_len = 16000 * 4
            if len(y) < target_len:
                y = np.pad(y, (0, target_len - len(y)))
            else:
                y = y[:target_len]
            mel = librosa.feature.melspectrogram(y=y, sr=16000, n_mels=self.n_mels)
            mel = librosa.power_to_db(mel, ref=np.max).astype(np.float32)
        # Chuẩn hóa kích thước về [n_mels, n_frames]
        if mel.shape[1] < self.n_frames:
            mel = np.pad(mel, ((0, 0), (0, self.n_frames - mel.shape[1])))
        else:
            mel = mel[:, : self.n_frames]
        # Chuẩn hóa mean/std đơn giản
        mean, std = float(mel.mean()), float(mel.std()) + 1e-6
        mel = (mel - mean) / std
        return mel.astype(np.float32)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        key = "mel_path" if "mel_path" in self.df.columns else ("filepath" if "filepath" in self.df.columns else "path")
        mel = self._load_mel(row[key])
        label_col = "label" if "label" in self.df.columns else "emotion"
        label = int(row[label_col])
        mel_t = torch.from_numpy(mel).unsqueeze(0)
        if self.transform is not None:
            mel_t = self.transform(mel_t)
        return {"mel": mel_t, "label": torch.tensor(label, dtype=torch.long)}
