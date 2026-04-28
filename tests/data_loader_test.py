from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from src.common.utils import print_green
from src.models.torch_data_loader import TorchDataLoader


class DataLoaderTest:
    @patch("src.models.data_loader.FeatureLoader")
    def test_load_all_splits(self, MockFeatureLoader):
        mock_instance = MockFeatureLoader.return_value

        mock_metadata = pd.DataFrame(
            {"audio_id": ["audio1", "audio2", "audio3", "audio4"], "label": ["bonafide", "spoof", "bonafide", "spoof"]}
        )
        mock_features = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8]], dtype=np.float32)

        mock_instance.load_data_split.return_value = (mock_metadata, mock_features)

        splits_to_test = ["train", "dev", "test"]

        for split in splits_to_test:
            dataset = TorchDataLoader(split=split, feat_suffix="dummy", device="cpu")

            assert len(dataset) == 4, f"Expected length 4, got {len(dataset)}"

            feat_0, label_0 = dataset[0]
            assert torch.allclose(
                feat_0, torch.tensor([0.1, 0.2], dtype=torch.float32)
            ), "Features of first element don't match"
            assert label_0.item() == 1.0, f"Expected label_0 to be 1 (bonafide), got {label_0.item()}"

            feat_1, label_1 = dataset[1]
            assert torch.allclose(
                feat_1, torch.tensor([0.3, 0.4], dtype=torch.float32)
            ), "Features of second element don't match"
            assert label_1.item() == 0.0, f"Expected label_1 to be 0 (spoof), got {label_1.item()}"

            mock_instance.load_data_split.assert_called_with(split_name=split)

            print_green(f"Dataset split '{split}' tested successfully!")


if __name__ == "__main__":
    tester = DataLoaderTest()
    tester.test_load_all_splits()
    print_green("\n>>> DataLoaderTest: All assertions passed successfully!")
