import numpy as np
import pandas as pd
import torch
import torchaudio

from src.common.constants import Constants as consts
from src.preprocessing.base_preprocessor import BasePreprocessor


class AudioSegmentator(BasePreprocessor):
    def __init__(self, overlap=2.0, max_duration=4.0, audio_type="spoof"):
        super().__init__(class_name=__class__.__name__)
        self.overlap_sec = overlap
        self.chunk_sec = max_duration
        self.audio_type = audio_type

    def _get_relevant_samples(self, wave_samples) -> list[torch.Tensor | int | float]:
        samples = wave_samples["array"]
        sr = wave_samples["sampling_rate"]
        duration = wave_samples["array"].shape[0] / sr
        tensor_samples = torch.tensor(samples, dtype=torch.float32).unsqueeze(0)
        return [tensor_samples, sr, duration]

    def _split_into_audio_slices(self, waveform, sr) -> tuple[list[np.ndarray], list[float]]:
        chunk_samples = int(self.chunk_sec * sr)
        stride_samples = int((self.chunk_sec - self.overlap_sec) * sr)

        chunks = []
        st_points = []
        steps = np.arange(0, waveform.shape[-1] - chunk_samples + 1, stride_samples)
        if steps[-1] + chunk_samples < waveform.shape[-1]:
            steps = np.append(steps, waveform.shape[-1] - chunk_samples)
        for start in steps:
            chunk = waveform.narrow(-1, start, chunk_samples).detach()
            chunks.append(chunk.squeeze().numpy())
            st_points.append(start / sr)

        return chunks, st_points

    def _resample(self, waveform, og_sr) -> torch.Tensor:
        resampler = torchaudio.transforms.Resample(orig_freq=og_sr, new_freq=consts.g_sample_rate)
        return resampler(waveform)

    def transform(self, data_set) -> tuple[pd.DataFrame, pd.DataFrame]:
        audio_segments_rows = []
        durations = []

        for record in data_set:
            waveform, sr, dur = self._get_relevant_samples(record["wav"])
            key_id = record["__key__"]

            if sr != consts.g_sample_rate:
                waveform = self._resample(waveform, sr)
                sr = consts.g_sample_rate
            durations.append({"key_id": key_id, "duration": dur})

            chunks, st_points = self._split_into_audio_slices(waveform, sr)
            for ch, stp in zip(chunks, st_points):
                audio_segments_rows.append(
                    {
                        "key_id": key_id,
                        "wave": ch,
                        "duration": ch.shape[-1] / sr,
                        "starting_point": stp,
                    }
                )

        audio_segments_df = pd.DataFrame(audio_segments_rows)
        durations_df = pd.DataFrame(durations)
        return audio_segments_df, durations_df
