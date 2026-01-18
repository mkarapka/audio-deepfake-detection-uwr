import numpy as np
import pandas as pd

from src.training.fft_baseline_classifier import FFTBaselineClassifier


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
        classifier = FFTBaselineClassifier(self.X_train, self.y_train, self.X_dev, self.meta_dev)
        model = RandomModel()

        all_preds = classifier._predict_all_records(model)
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
        classifier = FFTBaselineClassifier(self.X_train, self.y_train, self.X_dev, self.meta_dev)
        all_preds = classifier._predict_all_records(model)
        print("Obtained results:", all_preds.tolist())
        assert np.equal(all_preds, expected_res).all()
        print("test_majority_vote_correct_order passed.")


TestFFTBaselineClassifier().test_majority_vote()
TestFFTBaselineClassifier().test_majority_vote_correct_order()
