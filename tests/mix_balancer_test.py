import numpy as np
import pandas as pd

from src.preprocessing.data_balancers.mix_blancer import MixBalancer

COLUMN_LENGTH = 200
CONFIGS_RATIOS = [[0.5, 0.5], [0.75, 1.0], [1.0, 1.0]]


class MixBalancerTest:
    def __init__(self, undersample_ratio=0.75, oversample_ratio=1.0):
        self.balancer = MixBalancer(undersample_ratio=undersample_ratio, oversample_ratio=oversample_ratio)

    def _calculate_target_counts(self, metadata: pd.DataFrame, undersample_ratio: float, oversample_ratio: float):
        bonafide_count = metadata[metadata["target"] == "bonafide"].shape[0]

        S_new = int(bonafide_count / undersample_ratio)
        B_new = int(S_new * oversample_ratio)
        return B_new, S_new

    def test_mix_balancer(self):
        # Create a sample metadata DataFrame
        data = {
            "id": range(COLUMN_LENGTH),
            "target": np.random.choice(["bonafide", "spoof"], size=COLUMN_LENGTH, p=[0.3, 0.7]),
        }
        metadata = pd.DataFrame(data).set_index("id")

        # Create sample embeddings
        embeddings = np.random.rand(COLUMN_LENGTH, 128)  # 200 samples, 128-dimensional embeddings

        # Apply the MixBalancer
        balanced_metadata = self.balancer.transform(metadata)
        balanced_embeddings = embeddings[balanced_metadata.index]

        print("Original metadata count:")
        print(metadata["target"].value_counts())
        print("Balanced metadata count:")
        print(balanced_metadata["target"].value_counts())

        print("Shapes before balancing:")
        print(f"Metadata shape: {metadata.shape}, Embeddings shape: {embeddings.shape}")
        print("Shapes after balancing:")
        print(
            f"Balanced Metadata shape: {
                balanced_metadata.shape}, Balanced Embeddings shape: {
                balanced_embeddings.shape}"
        )
        # Check the results
        bonafide_count = balanced_metadata[balanced_metadata["target"] == "bonafide"].shape[0]
        spoof_count = balanced_metadata[balanced_metadata["target"] == "spoof"].shape[0]

        # assert bonafide_count == spoof_count, "The number of bonafide and spoof samples should be equal after balancing."
        assert (
            balanced_embeddings.shape[0] == balanced_metadata.shape[0]
        ), "Embeddings and metadata should have the same number of samples."

        target_B_new, target_S_new = self._calculate_target_counts(
            metadata, self.balancer.undersample_ratio, self.balancer.oversample_ratio
        )
        print("Expected bonafide and spoof count:", target_B_new, target_S_new)
        assert bonafide_count == target_B_new, "Bonafide count does not match the expected target after balancing."
        assert spoof_count == target_S_new, "Spoof count does not match the expected target after balancing."
        print("MixBalancer test passed. Balanced metadata and embeddings are consistent.")


def test_different_configs():
    for undersample_ratio, oversample_ratio in CONFIGS_RATIOS:
        print(f"\nTesting MixBalancer with undersample_ratio={undersample_ratio}, oversample_ratio={oversample_ratio}")
        balancer_test = MixBalancerTest(undersample_ratio=undersample_ratio, oversample_ratio=oversample_ratio)
        balancer_test.test_mix_balancer()


MixBalancerTest().test_mix_balancer()
test_different_configs()
