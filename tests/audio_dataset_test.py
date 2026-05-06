import numpy as np
import pandas as pd
import torch

from src.common.utils import print_green
from src.models.audio_dataset import AudioDataset


class AudioDatasetTest:
    def test_load_all_splits(self):
        metadata = pd.DataFrame(
            {
                "audio_id": ["audio1", "audio2", "audio3", "audio4"],
                "target": ["bonafide", "spoof", "bonafide", "spoof"],
            }
        )
        features = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8]], dtype=np.float32)

        dataset = AudioDataset(metadata=metadata, features=features, device="cpu")

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

        print_green("Dataset initialization tested successfully!")

    def test_with_dataloader(self):
        data_size = 100
        labels = [
            "bonafide",
            "spoof",
        ]
        metadata = pd.DataFrame(
            {
                "audio_id": [np.random.randint(1000) for _ in range(data_size)],
                "random_value_0": np.random.rand(data_size),
                "random_value_1": np.random.rand(data_size),
                "target": np.random.choice(labels, size=data_size, p=[0.3, 0.7]),
            }
        )
        features = np.array([[np.random.rand() for _ in range(28)] for _ in range(data_size)], dtype=np.float32)
        print(features.shape)

        dataset = AudioDataset(metadata=metadata, features=features, device="cpu")

        dataloader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=False)

        last_batch_size = data_size % 16

        for batch_idx, (feat_batch, label_batch) in enumerate(dataloader):
            if batch_idx == len(dataloader) - 1:
                assert feat_batch.shape == (
                    last_batch_size,
                    28,
                ), f"Expected feature batch shape ({last_batch_size}, 28), got {
                    feat_batch.shape}"
                assert label_batch.shape == (
                    last_batch_size,
                ), f"Expected label batch shape ({last_batch_size},), got {
                    label_batch.shape}"
            else:
                assert feat_batch.shape == (16, 28), f"Expected feature batch shape (16, 28), got {feat_batch.shape}"
                assert label_batch.shape == (16,), f"Expected label batch shape (16,), got {label_batch.shape}"
            assert set(label_batch.unique().tolist()).issubset(
                {0.0, 1.0}
            ), f"Labels should be binary (0 or 1), got {label_batch.unique().tolist()}"


if __name__ == "__main__":
    tester = AudioDatasetTest()
    tester.test_load_all_splits()
    tester.test_with_dataloader()
    print_green("\n>>> AudioDatasetTest: All assertions passed successfully!")
