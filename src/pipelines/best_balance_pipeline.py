from src.preprocessing.data_balancers.base_balancer import BaseBalancer
from src.preprocessing.data_balancers.mix_blancer import MixBalancer
from src.preprocessing.data_balancers.oversample_real_balancer import OversampleRealBalancer
from src.preprocessing.data_balancers.undersample_spoof_balancer import UndersampleSpoofBalancer
from src.preprocessing.feature_loader import FeatureLoader
import optuna
from sklearn.linear_model import LogisticRegression

class BestBalancePipeline:
    def __init__(self):
        self.results = {}
        
    def balance_dataset(self, metadata, balancer : BaseBalancer):
        balanced_metadata = balancer.transform(metadata=metadata)
        return balanced_metadata
    
    def oversample(self, metadata):
        oversampler = OversampleRealBalancer(real_to_spoof_ratio=0.5)
        return oversampler.transform(metadata=metadata)