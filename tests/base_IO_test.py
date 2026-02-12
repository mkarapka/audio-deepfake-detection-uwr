from pathlib import Path

from src.common.basic_functions import print_green
from src.common.constants import Constants as consts
from src.preprocessing.base_IO import BaseIO

TEST_SPLIT_DIR = consts.tests_data_dir / "splited_data"
FILE_NAME = "test_file"
FEAT_SUFFIX = "_feat"


class TestBaseIO:
    def test_create_file_path(self):
        base_io = BaseIO(
            class_name="TestBaseIO",
            file_name=FILE_NAME,
            feat_suffix=FEAT_SUFFIX,
            data_dir=consts.tests_data_dir,
            split_dir=TEST_SPLIT_DIR,
        )

        expected_file_path = consts.tests_data_dir / Path(f"{FILE_NAME}{FEAT_SUFFIX}.csv")
        actual_file_path = base_io._create_file_path(file_ext=".csv")
        print("Created file path:", actual_file_path)
        assert actual_file_path == expected_file_path, f"Expected {expected_file_path}, got {actual_file_path}"
        actual_file_path.touch()

        expected_file_path_split = TEST_SPLIT_DIR / Path(f"{FILE_NAME}{FEAT_SUFFIX}_split.csv")
        actual_file_path_split = base_io._create_file_path(file_ext=".csv", split_name="split")
        assert (
            actual_file_path_split == expected_file_path_split
        ), f"Expected {expected_file_path_split}, got {actual_file_path_split}"
        actual_file_path_split.touch()
        print_green("test_create_file_path passed successfully.")

    def not_existing_file_path(self):
        NON_EXISTING_FILE_NAME = "non_existing_file"
        base_io = BaseIO(
            class_name="TestBaseIO",
            file_name=NON_EXISTING_FILE_NAME,
            feat_suffix=FEAT_SUFFIX,
            data_dir=consts.tests_data_dir,
            split_dir=TEST_SPLIT_DIR,
        )
        try:
            base_io.create_read_file_path(file_ext=".csv", dir=consts.tests_data_dir)
        except Exception as e:
            print(f"Caught expected exception: {e}")
            print_green("not_existing_file_path test passed successfully.")
        else:
            assert False, "Expected FileNotFoundError was not raised."


TestBaseIO().test_create_file_path()
TestBaseIO().not_existing_file_path()
