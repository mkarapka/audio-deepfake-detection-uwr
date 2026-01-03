import numpy as np

from src.preprocessing.data_balancers.oversample_real_balancer import (
    OversampleRealBalancer,
)
from tests.dataset_balancer_test import TestDatasetBalancer


class OversampleRealBalancerTest:
    def __init__(self):
        self.balancer = OversampleRealBalancer()
        self.test_instance = TestDatasetBalancer(balancer=self.balancer, is_undersample=False)

    def test_same_seed(self):
        self.test_instance.test_with_various_seeds()

    def test_seed_consistency(self):
        self.test_instance.test_seed_consistency()

    def test_different_ratios(self, ratio_list=np.linspace(0.5, 1.0, 5).tolist()):
        self.test_instance.test_different_ratios(ratio_list)


OversampleRealBalancerTest().test_same_seed()
OversampleRealBalancerTest().test_seed_consistency()
OversampleRealBalancerTest().test_different_ratios()
