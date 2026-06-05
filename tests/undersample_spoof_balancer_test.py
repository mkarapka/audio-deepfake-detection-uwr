import numpy as np
import pandas as pd

from src.preprocessing.data_balancers.undersample_spoof_balancer import (
    UndersampleSpoofBalancer,
)
from tests.dataset_balancer_test import TestDatasetBalancer


class UndersampleSpoofBalancerTest:
    def __init__(self):
        self.balancer = UndersampleSpoofBalancer()
        self.test_instance = TestDatasetBalancer(balancer=self.balancer, is_undersample=True)

    def test_same_seed(self):
        self.test_instance.test_with_various_seeds()

    def test_seed_consistency(self):
        self.test_instance.test_seed_consistency()

    def test_different_ratios(self, ratio_list=np.linspace(0.5, 1.0, 5).tolist()):
        self.test_instance.test_different_ratios(ratio_list)

    def test_reduce_bonafide_when_spoof_is_minority(self):
        metadata = pd.DataFrame(
            {
                "feature1": list(range(100)),
                "target": ["bonafide"] * 80 + ["spoof"] * 20,
            }
        )
        features = np.random.rand(100, 8)

        balanced_metadata, balanced_features = self.balancer.transform(
            metadata=metadata,
            features=features,
            reduce_spoof=True,
        )

        bonafide_count = (balanced_metadata["target"] == "bonafide").sum()
        spoof_count = (balanced_metadata["target"] == "spoof").sum()

        assert bonafide_count == spoof_count == 20
        assert balanced_metadata.shape[0] == 40
        assert balanced_features.shape[0] == 40


UndersampleSpoofBalancerTest().test_same_seed()
UndersampleSpoofBalancerTest().test_seed_consistency()
UndersampleSpoofBalancerTest().test_different_ratios()
UndersampleSpoofBalancerTest().test_reduce_bonafide_when_spoof_is_minority()
