import numpy as np
import pandas as pd
import torch

from src.common.basic_functions import print_green
from src.common.constants import Constants as consts
from src.models.base_model import BaseModel

np.random.seed(27)
y_pred = np.random.choice([0, 1], 20)
buckets = np.array([0, 1, 2, 3])
y_preds_buckets = np.random.choice(buckets, size=len(y_pred))


class BaseModelTest:
    def __init__(self):
        self.model = None

    def _init_model(self):
        class TestModel(BaseModel):
            def predict(self, X, audio_ids=None):
                return X

            def load(self, model_name: str, ext: str, sub_dir: str = None):
                return None

            def save(self, model_name: str, ext: str, sub_dir: str = None):
                return None

        self.model = TestModel(class_name="TestModel")

    def check_not_implemented_methods_in_abstract_class_error(self):
        try:
            self.model = BaseModel(class_name="TestModel")
            assert False, "Expected TypeError for instantiating abstract class, but none was raised."
        except TypeError as e:
            print(f"Expected error caught for instantiation: {e}")

    def test_majority_voting(self):
        self._init_model()
        class DummyModel:
            def predict(self, X):
                return X

        self.model.model = DummyModel
        majoirity_voted_preds = self.model.majority_voting(y_pred=y_pred, audio_ids=pd.Series(y_preds_buckets))
        print(f"Original Predictions:       {y_pred}")
        print(f"Bucket Assignments:         {y_preds_buckets}")
        print(f"Majority Voted Predictions: {majoirity_voted_preds}")

        for b in buckets:
            mask = y_preds_buckets == b
            bucket_preds = y_pred[mask]
            majority_vote = int(bucket_preds.mean() >= 0.5)
            assert np.all(majoirity_voted_preds[mask] == majority_vote)

    def test_to_numpy(self):
        self._init_model()
        tensor_data = torch.tensor([1, 2, 3])
        numpy_data = self.model._to_numpy(tensor_data)
        print(f"Tensor data: {tensor_data}, Numpy data: {numpy_data}")
        assert isinstance(numpy_data, np.ndarray)
        assert np.array_equal(numpy_data, tensor_data.numpy())

    def test_get_model_file_path(self):
        self._init_model()
        MODEL_NAME = "TestDummyModel"

        self.model.models_dir = consts.tests_data_dir / "models"
        if not self.model.models_dir.exists():
            self.model.models_dir.mkdir(parents=True, exist_ok=True)

        expected_path = self.model.models_dir / f"{MODEL_NAME}.joblib"
        expected_path.touch()

        actual_path = self.model._get_model_file_path(model_name=MODEL_NAME, ext="joblib")
        print(f"Expected model file path: {expected_path}, Actual model file path: {actual_path}")
        assert actual_path == expected_path

    def test_get_model_file_path_not_found(self):
        self._init_model()
        MODEL_NAME = "NonExistentModel"
        try:
            self.model._get_model_file_path(model_name=MODEL_NAME, ext="joblib")
            assert False, "Expected an error for non-existent model file, but none was raised."
        except Exception as e:
            print(f"Expected error caught: {e}")

    def test_get_model_file_path_with_subdir(self):
        self._init_model()
        MODEL_NAME = "TestDummyModelWithSubdir"
        SUB_DIR = "subdir"

        self.model.models_dir = consts.tests_data_dir / "models"
        sub_dir_path = self.model.models_dir / SUB_DIR
        if not sub_dir_path.exists():
            sub_dir_path.mkdir(parents=True, exist_ok=True)

        expected_path = sub_dir_path / f"{MODEL_NAME}.joblib"
        expected_path.touch()

        actual_path = self.model._get_model_file_path(model_name=MODEL_NAME, ext="joblib", sub_dir=SUB_DIR)
        print(f"Expected model file path: {expected_path}, Actual model file path: {actual_path}")
        assert actual_path == expected_path

BaseModelTest().test_majority_voting()
BaseModelTest().test_to_numpy()
BaseModelTest().test_get_model_file_path()
BaseModelTest().test_get_model_file_path_not_found()
BaseModelTest().test_get_model_file_path_with_subdir()
print_green("All BaseModel tests passed successfully!")
