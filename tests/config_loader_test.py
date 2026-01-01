from src.common.constants import Constants as consts
from src.preprocessing.config_loader import ConfigLoader

DEV_SIZE = 3807
TEST_SIZE = 3769

FIRST_SPEAKER_ID_DEV = 10214
FIRST_SP_ROW_NO_DEV = 1

FIRST_SPEAKER_ID_TEST = 10226
FIRST_SP_ROW_NO_TEST = 0

SECOND_SPEAKER_ID_DEV = 10218
SECOND_SP_ROW_NO_DEV = 30


class TestConfigLoader:
    def test_load_next_config(self):
        configs = ["mls-tts-bark", "mls-vocoders-melgan"]
        splits = ["dev", "test"]
        config_loader = ConfigLoader(consts.audeter_ds_path, configs, splits)

        loaded_configs = []
        datasets = []
        for dataset in config_loader.stream_next_config_dataset():
            loaded_configs.append((config_loader.get_current_config(), config_loader.get_current_split()))
            datasets.append(dataset)
            # Check that dataset is not None
            assert dataset is not None

        expected_configs = [(cfg, sp) for cfg in configs for sp in splits]
        assert loaded_configs == expected_configs

        print("Loaded Configurations and Splits:")
        for cfg, sp in loaded_configs:
            print(f"Config: {cfg}, Split: {sp}")

        for i, (cfg, sp) in enumerate(expected_configs[2:4]):
            if sp == "dev":
                expected_size = DEV_SIZE
            else:
                expected_size = TEST_SIZE

            records_amount = 0
            for _ in datasets[i]:
                records_amount += 1
            assert records_amount == expected_size
            print(f"Dataset for Config: {cfg}, Split: {sp} has {records_amount} records - passed size check.")

    def check_if_mls_bonafide_is_loaded(self):
        configs = [None]
        splits = ["dev"]
        config_loader = ConfigLoader(consts.mls_eng_ds_path, configs, splits)

        dataset = next(config_loader.stream_next_config_dataset())
        assert dataset is not None

        print(f"Loaded dataset for config: {consts.mls_eng_config} - passed existence check.")

    def check_current_config_for_mls_bonafide(self):
        configs = [None]
        splits = ["dev"]
        config_loader = ConfigLoader(consts.mls_eng_ds_path, configs, splits)

        _ = next(config_loader.stream_next_config_dataset())
        current_config = config_loader.get_current_config()
        assert current_config == consts.mls_eng_config

        print(f"Current config is {current_config} - passed config name check.")

    def check_load_speakers_ids(self):
        configs = [None]
        splits = ["dev", "test"]
        config_loader = ConfigLoader(consts.mls_eng_ds_path, configs, splits)

        speakers_ids_df = config_loader.load_speakers_ids()

        print("Loaded speakers IDs DataFrame:")
        print(speakers_ids_df.head())
        print(speakers_ids_df.loc[FIRST_SP_ROW_NO_DEV, "dev"])
        print(speakers_ids_df.loc[SECOND_SP_ROW_NO_DEV, "dev"])
        print(speakers_ids_df.loc[FIRST_SP_ROW_NO_TEST, "test"])

        assert speakers_ids_df is not None
        assert not speakers_ids_df.empty
        assert "dev" in speakers_ids_df.columns
        assert "test" in speakers_ids_df.columns

        # Check number of unique speaker IDs
        assert speakers_ids_df.loc[FIRST_SP_ROW_NO_DEV, "dev"] == FIRST_SPEAKER_ID_DEV
        assert speakers_ids_df.loc[FIRST_SP_ROW_NO_TEST, "test"] == FIRST_SPEAKER_ID_TEST
        assert speakers_ids_df.loc[SECOND_SP_ROW_NO_DEV, "dev"] == SECOND_SPEAKER_ID_DEV
        assert speakers_ids_df.loc[SECOND_SP_ROW_NO_DEV - 1, "dev"] == FIRST_SPEAKER_ID_DEV

        # Check for padding in 'test' split
        assert speakers_ids_df.loc[TEST_SIZE, "test"] == -1  # Padding check
        assert speakers_ids_df.loc[DEV_SIZE - 1, "test"] == -1
        assert speakers_ids_df.loc[DEV_SIZE - 1, "dev"] != -1

        # Check DataFrame shape
        assert speakers_ids_df.shape[0] == DEV_SIZE  # Number of rows check
        assert speakers_ids_df.shape[1] == 2  # Number of columns check


TestConfigLoader().test_load_next_config()
TestConfigLoader().check_if_mls_bonafide_is_loaded()
TestConfigLoader().check_current_config_for_mls_bonafide()
TestConfigLoader().check_load_speakers_ids()
