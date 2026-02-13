import numpy as np
import pandas as pd
import torch

from src.models.base_model import BaseModel

np.random.seed(27)
y_preds = np.random.choice([0, 1], 20)
buckets = np.array([0, 1, 2, 3])
y_preds_buckets = np.random.choice(buckets, size=len(y_preds))


class BaseModelTest:
    def __init__(self):
        self.model = BaseModel(class_name="TestModel")

    def test_iterate_records(self):
        print(y_preds)
        print(y_preds_buckets)
        for record_preds, mask in self.model.iterate_records(uq_audio_ids=pd.Series(y_preds_buckets), y_preds=y_preds):
            mask_np = mask.to_numpy()
            curr_bucket = y_preds_buckets[mask_np][0]
            print(f"Bucket {curr_bucket}: {record_preds}")

            assert np.all(y_preds_buckets[mask_np] == curr_bucket)
            np.testing.assert_array_equal(record_preds, y_preds[mask_np])

    def test_majority_vote(self):
        y_preds = np.array([0, 0, 1, 1, 1])
        vote = self.model._majority_vote(y_preds)
        print(f"Predictions: {y_preds}, Majority Vote: {vote}")
        assert vote == 1

        y_preds = np.array([0, 0, 0, 1, 1])
        vote = self.model._majority_vote(y_preds)
        print(f"Predictions: {y_preds}, Majority Vote: {vote}")
        assert vote == 0

    def test_majority_voting(self):
        class DummyModel:
            def predict(self, X):
                return X

        majoirity_voted_preds = self.model.majority_voting(
            model=DummyModel(), X=y_preds, audio_ids=pd.Series(y_preds_buckets)
        )
        print(f"Original Predictions:       {y_preds}")
        print(f"Bucket Assignments:         {y_preds_buckets}")
        print(f"Majority Voted Predictions: {majoirity_voted_preds}")

        for b in buckets:
            mask = y_preds_buckets == b
            bucket_preds = y_preds[mask]
            majority_vote = self.model._majority_vote(bucket_preds)
            assert np.all(majoirity_voted_preds[mask] == majority_vote)

    def test_to_numpy(self):
        tensor_data = torch.tensor([1, 2, 3])
        numpy_data = self.model._to_numpy(tensor_data)
        print(f"Tensor data: {tensor_data}, Numpy data: {numpy_data}")
        assert isinstance(numpy_data, np.ndarray)
        assert np.array_equal(numpy_data, tensor_data.numpy())


BaseModelTest().test_iterate_records()
BaseModelTest().test_majority_vote()
BaseModelTest().test_majority_voting()
BaseModelTest().test_to_numpy()
print("All BaseModel tests passed successfully!")
