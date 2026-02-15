import numpy as np
import pandas as pd

ROWS_NO = 500
LABELS = ["bonafide", "spoof"]


class TestDatasetBalancer:
    def __init__(self, balancer=None, is_undersample=True):
        self.balancer = balancer
        self.is_undersample = is_undersample

    def _calculate_target_length_for_undersampling(self, metadata: pd.DataFrame, target_col: str, ratio: float):
        bonafide_count = np.sum(metadata[target_col] == "bonafide")

        B_new = bonafide_count
        S_new = int(B_new / ratio)
        target_length = B_new + S_new
        return target_length

    def _calculate_target_length_for_oversampling(self, metadata: pd.DataFrame, target_col: str, ratio: float):
        spoof_count = np.sum(metadata[target_col] == "spoof")

        S_new = spoof_count
        B_new = int(ratio * S_new)
        target_length = B_new + S_new
        return target_length

    def test_balance(self, total_rows=ROWS_NO, sample_labels=LABELS, seed=42, real_to_spoof_ratio=1.0):
        np.random.seed(seed)
        TARGET_COL = "target"
        metadata = pd.DataFrame(
            {
                "feature1": list(range(1, total_rows + 1)),
                "feature2": list(range(total_rows, 0, -1)),
                f"{TARGET_COL}": [np.random.choice(sample_labels, p=[0.3, 0.7]) for _ in range(total_rows)],
            }
        )
        if self.is_undersample:
            target_df_length = self._calculate_target_length_for_undersampling(
                metadata, TARGET_COL, real_to_spoof_ratio
            )
        else:
            target_df_length = self._calculate_target_length_for_oversampling(metadata, TARGET_COL, real_to_spoof_ratio)
        print("Tartget df length:", target_df_length)
        print("Target Column", metadata[f"{TARGET_COL}"].value_counts().to_dict())

        embeddings = np.random.rand(total_rows, 768)  # Example embeddings with 768 dimensions
        balanced_metadata, balanced_embeddings = self.balancer.transform(metadata, embeddings)

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

        # Test index consistency - sprawdź czy embeddingi odpowiadają metadata przez indeksy
        for i, idx in enumerate(self.balancer.previous_index):
            original_embedding = embeddings[idx]
            balanced_embedding = balanced_embeddings[i]
            assert np.array_equal(
                original_embedding, balanced_embedding
            ), f"Embedding at balanced position {i} should match original embedding at index {idx}"
            # Dodatkowo sprawdź czy feature1 się zgadza (feature1 = index + 1 w oryginalnych danych)
            expected_feature1 = idx + 1
            row = metadata.iloc[idx]
            assert (
                row["feature1"] == expected_feature1
            ), f"Metadata feature1 should match original data: expected {expected_feature1}, got {row['feature1']}"

        # Check that the balancing worked correctly for metadata
        expected_ratio = bonafide_count / spoof_count if spoof_count > 0 else 0
        print(
            f"Bonafide count: {bonafide_count}, Spoof count: {spoof_count}, Ratio B/S: {
                expected_ratio:.4f}, Expected: {real_to_spoof_ratio}"
        )
        assert (
            abs(expected_ratio - real_to_spoof_ratio) < 0.01
        ), f"The ratio of bonafide to spoof should be {real_to_spoof_ratio}, but got {expected_ratio:.4f}"
        print(
            "Balanced metadata size:",
            balanced_metadata.shape[0],
            "Target size:",
            target_df_length,
        )
        assert (
            balanced_metadata.shape[0] == target_df_length
        ), f"The balanced dataset has {
            balanced_metadata.shape[0]} records but should contain {target_df_length} records."

    def test_with_various_seeds(self, seed_list=[0, 1, 42, 100, 999]):
        for seed in seed_list:
            print(f"Testing with seed: {seed}")
            self.test_balance(total_rows=ROWS_NO, sample_labels=LABELS, seed=seed)

    def test_seed_consistency(self, seed=42):
        np.random.seed(seed)
        TARGET_COL = "target"
        metadata = pd.DataFrame(
            {
                "feature1": list(range(1, ROWS_NO + 1)),
                "feature2": list(range(ROWS_NO, 0, -1)),
                f"{TARGET_COL}": [np.random.choice(LABELS, p=[0.3, 0.7]) for _ in range(ROWS_NO)],
            }
        )
        embeddings = np.random.rand(ROWS_NO, 768)  # Example embeddings with 768 dimensions

        balanced_metadata1, balanced_embeddings1 = self.balancer.transform(metadata, embeddings)
        balanced_metadata2, balanced_embeddings2 = self.balancer.transform(metadata, embeddings)

        assert balanced_metadata1.equals(balanced_metadata2), "Balanced metadata should be the same for the same seed."
        assert np.array_equal(
            balanced_embeddings1, balanced_embeddings2
        ), "Balanced embeddings should be the same for the same seed."

    def test_different_ratios(self, ratio_list=[0.5, 0.75, 1.0]):
        for ratio in ratio_list:
            print(f"Testing with ratio: {ratio}")
            if hasattr(self.balancer, "real_to_spoof_ratio"):
                self.balancer.real_to_spoof_ratio = ratio
            self.test_balance(
                total_rows=ROWS_NO,
                sample_labels=LABELS,
                seed=42,
                real_to_spoof_ratio=ratio,
            )
