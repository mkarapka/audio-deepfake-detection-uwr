import warnings

import pandas as pd
import torch
from transformers import Wav2Vec2FeatureExtractor, WavLMModel

from preprocessing.base_preprocessor import BasePreprocessor


class WavLmExtractor(BasePreprocessor):
    def __init__(self, pretrained_model_name="microsoft/wavlm-base"):
        warnings.filterwarnings("ignore", message=".*key_padding_mask and attn_mask.*")

        self.pretrained_model_name = pretrained_model_name
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(self.pretrained_model_name)
        self.model = WavLMModel.from_pretrained(self.pretrained_model_name)

    def transform(self, data: pd.DataFrame, sample_rate):
        sr = sample_rate
        embeddings = []
        for _, record in data.iterrows():
            waveform = record["wave"]
            inputs = self.feature_extractor(waveform, sampling_rate=sr, return_tensors="pt")

            with torch.no_grad():
                outputs = self.model(**inputs)
                embedding = outputs.last_hidden_state.mean(dim=1)
                embeddings.append(embedding.squeeze().numpy())

        data["wave"] = embeddings
        return data
