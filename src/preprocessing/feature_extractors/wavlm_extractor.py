import warnings

import numpy as np
import torch
from transformers import Wav2Vec2FeatureExtractor, WavLMModel

from src.common.basic_functions import get_device
from src.common.constants import Constants as consts
from src.preprocessing.feature_extractors.base_feature_extractor import (
    BaseFeatureExtractor,
)


class WavLmExtractor(BaseFeatureExtractor):
    def __init__(self, batch_size=8, pretrained_model_name=consts.wavlm_base_plus_name, sample_rate=16_000):
        warnings.filterwarnings("ignore", message=".*key_padding_mask and attn_mask.*")
        super().__init__(class_name=__class__.__name__)

        self.device = get_device()
        self.batch_size = batch_size
        self.sample_rate = sample_rate
        self.pretrained_model_name = pretrained_model_name
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(self.pretrained_model_name)
        self.model = WavLMModel.from_pretrained(self.pretrained_model_name).to(self.device)
        self.print_percent = 0.1

    def _print_percent_of_completed_records(self, i, rows_size):
        step = max(1, int(rows_size // self.batch_size * self.print_percent))
        if (i // self.batch_size) % step == 0:
            print(f"WavLM Extractor: Processed {i / rows_size:.2%} records.")

    def transform(self, wave_segments: np.ndarray) -> torch.Tensor:
        rows_size = wave_segments.shape[0]
        sr = self.sample_rate
        all_embeddings = []

        for i in range(0, wave_segments.shape[0], self.batch_size):
            batch_wave_seq = wave_segments[i : i + self.batch_size]
            inputs = self.feature_extractor(batch_wave_seq, sampling_rate=sr, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model(**inputs)
                embeddings = outputs.last_hidden_state.mean(dim=1)
                all_embeddings.append(embeddings)

            self._print_percent_of_completed_records(i, rows_size)

        full_embeddings = torch.cat(all_embeddings, dim=0).cpu().numpy()
        return full_embeddings
