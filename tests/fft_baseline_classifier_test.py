import numpy as np
import pandas as pd

from src.models.fft_baseline_classifier import FFTBaselineClassifier


class RandomModel:
    def predict(self, X):
        np.random.seed(42)
        return np.random.randint(0, 2, size=X.shape[0])


class RandomMoreLikelyRealModel:
    def predict(self, X):
        np.random.seed(42)
        return np.random.choice([0, 1], size=X.shape[0], p=[0.4, 0.6])


class DummyModel:
    def __init__(self, expected_results):
        self.expected_results = expected_results

    def predict(self, _):
        return np.array(self.expected_results)


class TestFFTBaselineClassifier:
    def __init__(self, size=10):
        class_map = {0: "bonafide", 1: "spoof", 2: "bonafide"}
        audio_ids = [0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 2, 2, 2]
        length = len(audio_ids)
        size = length
        example_dev_meta = {
            "unique_audio_id": audio_ids,
            "target": [class_map[id] for id in audio_ids],
        }
        self.X_train = np.random.rand(size, 10)
        self.y_train = np.array(example_dev_meta["target"])
        self.X_dev = np.random.rand(size, 10)
        self.meta_dev = pd.DataFrame(example_dev_meta)

    def test_majority_vote(self):
        np.random.seed(42)
        self.meta_dev.copy()
        classifier = FFTBaselineClassifier(
            is_chunk_prediction=True, dev_uq_audio_ids=self.meta_dev["unique_audio_id"]
        )
        model = RandomModel()
        classifier.eval_model = model

        all_preds = classifier.predict(self.X_dev)
        test_preds = model.predict(self.X_dev)
        x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        print(model.predict(x))
        print(model.predict(x))

        for _, id in enumerate(self.meta_dev["unique_audio_id"].unique()):
            record_mask = self.meta_dev["unique_audio_id"] == id
            record_preds = test_preds[record_mask]
            vote = int((record_preds.mean() >= 0.5))
            print(all_preds[record_mask], vote)
            print(record_preds)
            assert np.mean(all_preds[record_mask]) == vote

        assert len(all_preds) == self.meta_dev.shape[0]

        print("test_majority_vote passed.")

    def test_majority_vote_correct_order(self):
        mapper = {0: 1, 1: 0, 2: 1}
        expected_res = [mapper[i] for i in self.meta_dev["unique_audio_id"]]
        print("Expected results:", expected_res)
        model = DummyModel(expected_results=expected_res)
        classifier = FFTBaselineClassifier(
            is_chunk_prediction=True, dev_uq_audio_ids=self.meta_dev["unique_audio_id"]
        )
        classifier.eval_model = model
        all_preds = classifier.predict(self.X_dev)
        print("Obtained results:", all_preds.tolist())
        assert np.equal(all_preds, expected_res).all()
        print("test_majority_vote_correct_order passed.")

    def test_normal_predict(self):
        model = RandomMoreLikelyRealModel()
        classifier = FFTBaselineClassifier(is_chunk_prediction=False)
        classifier.eval_model = model
        all_preds = classifier.predict(self.X_dev)
        expected_preds = model.predict(self.X_dev)
        assert np.equal(all_preds, expected_preds).all()
        print("test_normal_predict passed.")

    def test_get_model(self):
        classifier = FFTBaselineClassifier(is_chunk_prediction=False)
        params = {
            "max_depth": 5,
            "learning_rate": 0.1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "gamma": 0.1,
            "n_estimators": 200,
            "scale_pos_weight": 1.0,
        }
        model = classifier.get_model(params)
        assert model.max_depth == 5
        assert model.learning_rate == 0.1
        assert model.subsample == 0.8
        assert model.colsample_bytree == 0.8
        assert model.gamma == 0.1
        assert model.n_estimators == 200
        assert model.scale_pos_weight == 1.0
        print("test_get_model passed.")

    def test_optuna_fit(self):
        classifier = FFTBaselineClassifier(is_chunk_prediction=False)
        classifier.optuna_fit(
            n_trials=2,
            X_train=self.X_train,
            y_train=self.y_train,
            X_dev=self.X_dev,
            y_dev=self.meta_dev["target"].values,
        )
        best_params = classifier.get_best_params()
        assert best_params is not None
        print("test_optuna_fit passed.")

    def test_fit_and_predict(self, is_chunk_prediction: bool = False):
        classifier = FFTBaselineClassifier(is_chunk_prediction=is_chunk_prediction)
        params = {
            "max_depth": 4,
            "learning_rate": 0.05,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "gamma": 0.2,
            "n_estimators": 150,
            "scale_pos_weight": 1.0,
        }
        classifier.set_model(params)
        classifier.fit(self.X_train, self.y_train)
        predictions = classifier.predict(self.X_dev)
        assert len(predictions) == self.X_dev.shape[0]
        print("test_fit_and_predict passed.")


TestFFTBaselineClassifier().test_majority_vote()
TestFFTBaselineClassifier().test_majority_vote_correct_order()
TestFFTBaselineClassifier().test_normal_predict()
TestFFTBaselineClassifier().test_get_model()
TestFFTBaselineClassifier().test_optuna_fit()
TestFFTBaselineClassifier().test_fit_and_predict(is_chunk_prediction=False)
TestFFTBaselineClassifier().test_fit_and_predict(is_chunk_prediction=True)
