from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.common.constants import Constants as consts
from src.common.utils import print_green
from src.models.xgboost_classifier import XGBoostClassifier


class XGBoostClassifierTest:
    @staticmethod
    def _make_mock_model():
        model = MagicMock()
        model.predict = MagicMock()
        model.save_model = MagicMock()
        model.load_model = MagicMock()
        return model

    @staticmethod
    def _test_model_path() -> str:
        model_dir = consts.tests_data_dir / "models"
        model_dir.mkdir(parents=True, exist_ok=True)
        return str(model_dir / "xgboost_test_model.json")

    def test_predict_without_audio_ids_returns_thresholded_labels(self):
        model = self._make_mock_model()
        model.predict.return_value = np.array([0.1, 0.7, 0.4])
        classifier = XGBoostClassifier(model=model, device="cpu")

        X = np.array([[1.0], [2.0], [3.0]])
        y_pred = classifier.predict(X=X, threshold=0.5)

        assert np.array_equal(y_pred, np.array([0, 1, 0]))

    def test_predict_with_audio_ids_applies_majority_voting(self):
        model = self._make_mock_model()
        model.predict.return_value = np.array([0.6, 0.2, 0.8])
        classifier = XGBoostClassifier(model=model, device="cpu")

        X = np.array([[1.0], [2.0], [3.0]])
        audio_ids = ["a", "a", "b"]
        y_pred = classifier.predict(X=X, audio_ids=audio_ids, threshold=0.5)

        assert np.array_equal(y_pred, np.array([0, 0, 1]))

    def test_save_with_invalid_extension_raises_error(self):
        model = self._make_mock_model()
        classifier = XGBoostClassifier(model=model, device="cpu")

        with pytest.raises(Exception, match="Invalid file format"):
            classifier.save("model.bin")

    def test_load_with_invalid_extension_raises_error(self):
        model = self._make_mock_model()
        classifier = XGBoostClassifier(model=model, device="cpu")

        with pytest.raises(Exception, match="Invalid file format"):
            classifier.load("model.bin")

    def test_save_with_json_extension_calls_model_save(self):
        model = self._make_mock_model()
        classifier = XGBoostClassifier(model=model, device="cpu")
        model_path = self._test_model_path()

        classifier.save(model_path)

        model.save_model.assert_called_once_with(model_path)

    def test_load_with_json_extension_replaces_model_and_loads_weights(self):
        initial_model = self._make_mock_model()
        loaded_model = self._make_mock_model()
        classifier = XGBoostClassifier(model=initial_model, device="cpu")
        model_path = self._test_model_path()

        with patch("src.models.xgboost_classifier.xgb.Booster", return_value=loaded_model) as patched_ctor:
            classifier.load(model_path)

        patched_ctor.assert_called_once_with()
        loaded_model.load_model.assert_called_once_with(model_path)
        assert classifier.model is loaded_model


tester = XGBoostClassifierTest()
tester.test_predict_without_audio_ids_returns_thresholded_labels()
tester.test_predict_with_audio_ids_applies_majority_voting()
tester.test_save_with_invalid_extension_raises_error()
tester.test_load_with_invalid_extension_raises_error()
tester.test_save_with_json_extension_calls_model_save()
tester.test_load_with_json_extension_replaces_model_and_loads_weights()
print_green("All XGBoostClassifier tests passed successfully!")
