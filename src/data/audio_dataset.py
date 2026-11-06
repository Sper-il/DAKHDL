"""Dataset nhánh tiếng: WAV 16kHz mono -> Mel-spectrogram 128x128 (hỗ trợ Waveform Augmentation chống overfit)."""
from __future__ import annotations
from pathlib import Path
import random
import numpy as np
import torch
from torch.utils.data import Dataset

EMOTIONS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]


def _augment_waveform(y: np.ndarray, sr: int = 16000) -> np.ndarray:
    """Biến đổi sóng âm lúc train: dịch thời gian (±0.35s), đổi âm lượng (±5dB), thêm nhiễu trắng nhẹ."""
    out = y.copy()
    # 1. Dịch thời gian ngẫu nhiên (Time shift)
    if random.random() < 0.6:
        max_shift = int(0.35 * sr)
        shift = random.randint(-max_shift, max_shift)
        out = np.roll(out, shift)
    # 2. Thay đổi biên độ ngẫu nhiên (Random Gain ±5dB)
    if random.random() < 0.6:
        gain_db = random.uniform(-5.0, 5.0)
        out = out * (10.0 ** (gain_db / 20.0))
    # 3. Thêm nhiễu nền Gaussian nhẹ (SNR 18..30 dB)
    if random.random() < 0.5:
        sig_power = float(np.mean(out ** 2)) + 1e-8
        snr_db = random.uniform(18.0, 30.0)
        noise_power = sig_power / (10.0 ** (snr_db / 10.0))
        noise = np.random.normal(0.0, np.sqrt(noise_power), size=out.shape).astype(np.float32)
        out = out + noise
    return np.clip(out, -1.0, 1.0).astype(np.float32)


class AudioDataset(Dataset):
    """Đọc file npy mel đã xử lý hoặc WAV gốc rồi trích mel on-the-fly.

    Output mỗi mẫu: ``{'mel': tensor [1,128,128], 'label': int}``.
    """

    def __init__(
        self,
        csv_path: str | Path,
        transform=None,
        n_mels: int = 128,
        n_frames: int = 128,
        augment_wave: bool = False,
    ):
        import pandas as pd
        self.df = pd.read_csv(csv_path)
        self.transform = transform
        self.n_mels = n_mels
        self.n_frames = n_frames
        self.augment_wave = augment_wave

    def __len__(self) -> int:
        return len(self.df)

    def _load_mel(self, path: str) -> np.ndarray:
        p = str(path)
        if p.endswith(".npy"):
            mel = np.load(p).astype(np.float32)
        else:
            import librosa
            import soundfile as sf
            y, sr = sf.read(p, dtype="float32")
            if y.ndim > 1:
                y = y.mean(axis=1)
            if sr != 16000:
                y = librosa.resample(y, orig_sr=sr, target_sr=16000)
            target_len = 16000 * 4
            if len(y) < target_len:
                y = np.pad(y, (0, target_len - len(y)))
            else:
                y = y[:target_len]
            if self.augment_wave:
                y = _augment_waveform(y, sr=16000)
            mel = librosa.feature.melspectrogram(y=y, sr=16000, n_mels=self.n_mels)
            mel = librosa.power_to_db(mel, ref=np.max).astype(np.float32)

        if mel.shape[1] < self.n_frames:
            mel = np.pad(mel, ((0, 0), (0, self.n_frames - mel.shape[1])))
        else:
            mel = mel[:, : self.n_frames]
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
