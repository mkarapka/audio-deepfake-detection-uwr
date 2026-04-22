from src.models.base_model import BaseModel
import xgboost as xgb
from src.common.logger import raise_error_logger


class XGBoostClassifier(BaseModel):
    def __init__(self, model : xgb.XGBClassifier, device: str = None):
        super().__init__(class_name=self.__class__.__name__, device=device)
        self.model = model

    def predict(self, X, audio_ids=None, threshold=0.5):
        y_probs = self.model.predict_proba(X)[:, 1]
        if audio_ids:
            y_pred = self.majority_voting(y_probs=y_probs, audio_ids=audio_ids, threshold=threshold)
        else:
            y_pred = (y_probs >= threshold).astype(int)
        return y_pred

    def load(self, file_path: str):
        if not file_path.endswith(".json"):
            raise_error_logger(self.logger, f"Invalid file format for XGBoost model: {file_path}")
        self.model = xgb.XGBClassifier()
        self.model.load_model(file_path)

    def save(self, file_path: str):
        if not file_path.endswith(".json"):
            raise_error_logger(self.logger, f"Invalid file format for XGBoost model: {file_path}")
        self.model.save_model(file_path)
