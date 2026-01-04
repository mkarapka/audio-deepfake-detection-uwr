import numpy as np
import torch

from src.common.basic_functions import get_device
from src.preprocessing.feature_extractors.base_feature_extractor import (
    BaseFeatureExtractor,
)


class FFTExtractor(BaseFeatureExtractor):
    def __init__(self, window_size=1024, hop_length=512, batch_size=8):
        super().__init__(class_name=__class__.__name__)
        self.window_size = window_size
        self.hop_length = hop_length
        self.batch_size = batch_size
        self.device = get_device()

    def _frame_signal_batch(self, signals: torch.Tensor) -> torch.Tensor:
        _, T = signals.shape
        frames = []
        for start in range(0, T - self.window_size + 1, self.hop_length):
            frames.append(signals[:, start : start + self.window_size])

        return torch.stack(frames, dim=1)  # [B, n_frames, win]

    def _extract_features(self, signals: torch.Tensor) -> torch.Tensor:
        signals = signals.to(self.device)
        frames = self._frame_signal_batch(signals)

        window = torch.hann_window(self.window_size, device=self.device)
        frames_win = frames * window

        fft_complex = torch.fft.rfft(frames_win, dim=-1)

        mag = torch.abs(fft_complex)
        eps = 1e-10
        log_mag = torch.log(mag + eps)

        mean_feat = log_mag.mean(dim=1)
        std_feat = log_mag.std(dim=1)

        feat_vec = torch.cat([mean_feat, std_feat], dim=-1)
        return feat_vec

    def transform(self, wave_segments: np.ndarray) -> np.ndarray:
        rows_size = wave_segments.shape[0]
        all_features = []
        for i in range(0, wave_segments.shape[0], self.batch_size):
            batch_wave_seq = wave_segments[i : i + self.batch_size]
            batch_wave_tensor = torch.tensor(batch_wave_seq, dtype=torch.float32)

            with torch.no_grad():
                features = self._extract_features(batch_wave_tensor)
                all_features.append(features.cpu().numpy())

            step = max(1, int(rows_size // self.batch_size * 0.1))
            if (i // self.batch_size) % step == 0:
                print(f"FFT Extractor: Processed {i / rows_size:.2%} records.")

        full_features = np.vstack(all_features)
        return full_features
