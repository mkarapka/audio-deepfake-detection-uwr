from itertools import combinations

import numpy as np
import pandas as pd

from src.common.constants import Constants as consts
from src.preprocessing.config_loader import ConfigLoader
from src.preprocessing.dataset_spliter import DatasetSpliter


class TestDatasetSpliter:
    def __init__(self):
        self.spliter = DatasetSpliter(
            config=consts.basic_train_dev_test_config,
            speakers_ids=ConfigLoader(consts.audeter_ds_path, config=[None]).load_speakers_ids(),
            seed=42,
        )
        self.example_configs = [
            "mls-tts-yourtts",
            "mls-tts-bark",
            "mls-tts-fastspeech2",
        ]

    def get_random_data(self, data, records=10, seed=42, with_replacement=True):
        np.random.seed(seed)
        return np.random.choice(data, size=records, replace=with_replacement)

    def gen_speaker_ids_map(self, records_lst, ex_speakers_ids, seed=42):
        np.random.seed(seed)
        speakers_map = {}

        for r in records_lst:
            if r not in speakers_map:
                speakers_map[r] = np.random.choice(ex_speakers_ids, size=1)[0]
        return speakers_map

    def gen_speaker_ids(self, records_no, seed=42):
        ids_train, ids_dev, ids_test = self.spliter._get_train_dev_test_speakers_ids(
            self.spliter._get_unique_speakers_ids()
        )
        train_rd = int(records_no * 0.6)
        dev_test_rd = int(records_no * 0.2)
        ex_speakers_ids = np.hstack(
            [
                np.random.choice(ids_train, train_rd),
                np.random.choice(ids_dev, dev_test_rd),
                np.random.choice(ids_test, dev_test_rd),
            ]
        )
        return ex_speakers_ids

    def prepare_data(self, records_no=10):
        records_ids = self.get_random_data(list(range(0, 5)), records=records_no)
        ex_speakers_ids = self.gen_speaker_ids(records_no=records_no)
        speaker_ids_map = self.gen_speaker_ids_map(records_ids, ex_speakers_ids)

        data = {
            "config": self.get_random_data(self.example_configs, records=records_no),
            "split": self.get_random_data(["dev", "test"], records=records_no),
            "record_id": records_ids,
            "speaker_id": [speaker_ids_map[rid] for rid in records_ids],
            "target": self.get_random_data(["spoof", "bonafide"], records=records_no),
        }
        data = pd.DataFrame(data)
        data = data.sort_values(by=["config", "split", "record_id"]).reset_index(drop=True)

        example_embeddings = np.random.rand(records_no, 768)
        return data, example_embeddings

    def check_if_speakers_ids_set_is_disjoint(self, splits_data: dict[str, pd.DataFrame]):
        splits_combinations = combinations(splits_data.items(), r=2)
        for (l_name, l_set), (r_name, r_set) in splits_combinations:
            assert l_set.isdisjoint(r_set), f"Speakers IDs overlap between {l_name} and {r_name}"

    def test_unique_speakers_ids(self):
        uq_speakers_ids = self.spliter._get_unique_speakers_ids()
        assert len(set(uq_speakers_ids)) == 84

    def test_get_train_dev_test_speakers_ids(self):
        uq_speakers_ids = self.spliter._get_unique_speakers_ids()
        ids_train, ids_dev, ids_test = self.spliter._get_train_dev_test_speakers_ids(uq_speakers_ids)
        total_ids = len(ids_train) + len(ids_dev) + len(ids_test)
        assert total_ids == len(
            uq_speakers_ids
        ), "Total number of speakers IDs in splits does not match the unique speakers IDs!"
        print(
            "Size of train/dev/test splits:",
            len(ids_train),
            len(ids_dev),
            len(ids_test),
        )
        print(
            f"Percentages: {
                len(ids_train) /
                total_ids:.2f}, {
                len(ids_dev) /
                total_ids:.2f}, {
                len(ids_test) /
                total_ids:.2f}"
        )
        self.check_if_speakers_ids_set_is_disjoint(
            {
                "train": set(ids_train),
                "dev": set(ids_dev),
                "test": set(ids_test),
            }
        )

    def test_transform(self, records_no):
        data, example_embeddings = self.prepare_data(records_no=records_no)
        print("Original DataFrame:")
        print(data)

        (train, emb_train), (dev, emb_dev), (test, emb_test) = self.spliter.transform(data, example_embeddings)

        uq_speakers_ids = set(data["speaker_id"].unique())
        uq_speakers_ids_in_splits = {
            "train": set(train["speaker_id"].unique()),
            "dev": set(dev["speaker_id"].unique()),
            "test": set(test["speaker_id"].unique()),
        }

        assert len(uq_speakers_ids) == sum(len(s) for s in uq_speakers_ids_in_splits.values())
        self.check_if_speakers_ids_set_is_disjoint(uq_speakers_ids_in_splits)

        assert train.shape[0] > 0 and dev.shape[0] > 0 and test.shape[0] > 0, "At least one of the splits is empty!"

        assert (
            emb_train.shape[0] == train.shape[0]
            and emb_dev.shape[0] == dev.shape[0]
            and emb_test.shape[0] == test.shape[0]
        ), "Embeddings and metadata size mismatch in at least one split!"


TestDatasetSpliter().test_unique_speakers_ids()
TestDatasetSpliter().test_get_train_dev_test_speakers_ids()
TestDatasetSpliter().test_transform(30)
