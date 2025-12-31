from src.common.constants import Constants as consts
from src.preprocessing.config_loader import ConfigLoader

DEV_SIZE = 3807
TEST_SIZE = 3769


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


TestConfigLoader().test_load_next_config()
TestConfigLoader().check_if_mls_bonafide_is_loaded()
TestConfigLoader().check_current_config_for_mls_bonafide()
