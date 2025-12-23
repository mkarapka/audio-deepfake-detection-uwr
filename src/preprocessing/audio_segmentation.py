import torchaudio
import numpy as np
import pandas as pd
from src.common.constants import Constants as const


class AudioSegmentator:
    def __init__(self, overlap=2.0, max_duration=4.0):
        self.overlap_sec = overlap
        self.chunk_sec = max_duration

    def get_relevant_samples(self, wave_samples):
        samples = wave_samples.get_all_samples()
        return [samples.data, samples.sample_rate, samples.duration_seconds]

    def split_into_audio_slices(self, waveform, sr):
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

    def resample(self, waveform, og_sr):
        resampler = torchaudio.transforms.Resample(
            orig_freq=og_sr, new_freq=const.g_sample_rate
        )
        return resampler(waveform)

    def transform(self, data_set):
        audio_segments_rows = []
        durations = []

        for record in data_set:
            waveform, sr, dur = self.get_relevant_samples(record["wav"])
            key_id = record["__key__"]

            if sr != const.g_sample_rate:
                waveform = self.resample(waveform, sr)
            durations.append({"key_id": key_id, "duration": dur})

            chunks, st_points = self.split_into_audio_slices(waveform, sr)
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
