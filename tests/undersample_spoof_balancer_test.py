import numpy as np

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


UndersampleSpoofBalancerTest().test_same_seed()
UndersampleSpoofBalancerTest().test_seed_consistency()
UndersampleSpoofBalancerTest().test_different_ratios()
