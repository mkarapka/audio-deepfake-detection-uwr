import numpy as np
import pandas as pd

from src.training.record_iterator import RecordIterator


class TestRecordIterator:
    def __init__(self, unique_ids=50):
        self.unique_ids = unique_ids
        multiply = [3, 4, 5, 6]
        unique_audio_ids = np.hstack([np.random.choice(multiply) * [i] for i in range(self.unique_ids)]).tolist()
        self.metadata = pd.DataFrame({"unique_audio_id": unique_audio_ids, "feature": range(len(unique_audio_ids))})
        self.embeddings = np.random.rand(len(self.metadata), 10)

    def test_sample_fraction(self):
        iterator = RecordIterator()
        new_metadata = iterator.sample_fraction(self.metadata, 0.5)

        assert (
            self.metadata[self.metadata["unique_audio_id"].isin(new_metadata["unique_audio_id"])].shape[0]
            == new_metadata.shape[0]
        )
        assert self.embeddings[new_metadata.index].shape[0] == new_metadata.shape[0]
        print("test_sample_fraction passed.")

    def test_iterate_records(self):
        iterator = RecordIterator()
        count = 0
        for record_embeddings, mask in iterator.iterate_records(self.metadata, self.embeddings):
            record_metadata = self.metadata[mask]
            assert record_metadata.shape[0] == record_embeddings.shape[0]
            assert all(record_metadata["unique_audio_id"] == record_metadata.iloc[0]["unique_audio_id"])
            assert len(record_metadata) == np.sum(
                self.metadata["unique_audio_id"] == record_metadata.iloc[0]["unique_audio_id"]
            )
            count += 1
        assert count == self.unique_ids
        print("test_iterate_records passed.")


TestRecordIterator().test_sample_fraction()
TestRecordIterator().test_iterate_records()
