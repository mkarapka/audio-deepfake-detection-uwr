import numpy as np
import pandas as pd

from src.preprocessing.data_balancers.undersample_spoof_balancer import (
    UndersampleSpoofBalancer,
)

ROWS_NO = 500
LABELS = ["bonafide", "spoof"]


class TestUndersampleSpoofBalancer:
    def test_balance(self, total_rows=ROWS_NO, sample_labels=LABELS, seed=42):
        np.random.seed(seed)
        TARGET_COL = "target"
        metadata = pd.DataFrame(
            {
                "feature1": list(range(1, total_rows + 1)),
                "feature2": list(range(total_rows, 0, -1)),
                f"{TARGET_COL}": [np.random.choice(sample_labels, p=[0.3, 0.7]) for _ in range(total_rows)],
            }
        )
        target_df_length = np.sum(metadata[f"{TARGET_COL}"] == "bonafide") * 2
        print("Tartget df length:", target_df_length)
        print("Target Column", metadata[f"{TARGET_COL}"].value_counts().to_dict())

        embeddings = np.random.rand(total_rows, 768)  # Example embeddings with 768 dimensions
        balancer = UndersampleSpoofBalancer(seed=42)
        balanced_metadata, balanced_embeddings = balancer.transform(metadata, embeddings)

        bonafide_mask = balanced_metadata[f"{TARGET_COL}"] == "bonafide"
        spoof_mask = balanced_metadata[f"{TARGET_COL}"] == "spoof"

        bonafide_count = sum(bonafide_mask)
        spoof_count = sum(spoof_mask)

        bonafide_embeddings = balanced_embeddings[bonafide_mask]
        spoof_embeddings = balanced_embeddings[spoof_mask]

        # Check that the balancing worked correctly for embeddings
        assert balanced_embeddings.shape[0] == balanced_metadata.shape[0]
        assert (
            bonafide_embeddings.shape[0] == bonafide_count
        ), "The number of bonafide embeddings should match the number of bonafide metadata records."
        assert (
            spoof_embeddings.shape[0] == spoof_count
        ), "The number of spoof embeddings should match the number of spoof metadata records."

        concated_embeddings = np.vstack([bonafide_embeddings, spoof_embeddings])
        assert (
            concated_embeddings.shape[0] == balanced_embeddings.shape[0]
        ), "Embeddings should not have duplicates after balancing."
        assert (
            concated_embeddings.shape[1] == balanced_embeddings.shape[1]
        ), "Embeddings dimensionality should remain unchanged after balancing."
        assert (
            np.sort(concated_embeddings)[0:5].all() == np.sort(balanced_embeddings)[0:5].all()
        ), "Embeddings content should remain unchanged after balancing."

        # Check that the balancing worked correctly for metadata
        assert (
            bonafide_count == spoof_count
        ), "The number of bonafide and spoof samples should be equal after balancing."
        assert (
            balanced_metadata.shape[0] == target_df_length
        ), f"The balanced dataset should contain {target_df_length} records."

    def test_with_various_seeds(self):
        for seed in [0, 1, 42, 100, 999]:
            print(f"Testing with seed: {seed}")
            TestUndersampleSpoofBalancer().test_balance(total_rows=ROWS_NO, sample_labels=LABELS, seed=seed)


TestUndersampleSpoofBalancer().test_with_various_seeds()
