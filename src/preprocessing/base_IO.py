from pathlib import Path

from src.common.logger import raise_error_logger, setup_logger


class BaseIO:
    def __init__(
        self,
        class_name: str,
        file_name: str,
        feat_suffix: str,
        data_dir: Path,
        split_dir: Path,
    ):
        self.logger = setup_logger(class_name, log_to_console=True)
        if "_wavlm" == feat_suffix:
            self.logger.info("Using WavLM embeddings suffix")
        elif "_fft" == feat_suffix:
            self.logger.info("Using FFT embeddings suffix")
        else:
            self.logger.warning("Not matching feature suffix specified")

        self.full_file_name = file_name + feat_suffix
        self.data_dir = data_dir
        self.split_dir = split_dir

        if self.data_dir.exists() is False:
            self.data_dir.mkdir(parents=True, exist_ok=True)
        if self.split_dir.exists() is False:
            self.split_dir.mkdir(parents=True, exist_ok=True)

    def _create_file_path(self, file_ext: str, dir: str = None, split_name: str = None) -> Path:
        if dir is None:
            dir = self.data_dir
            if split_name is not None:
                dir = self.split_dir

        if split_name is not None:
            new_file_name = f"{self.full_file_name}_{split_name}{file_ext}"
        else:
            new_file_name = f"{self.full_file_name}{file_ext}"

        file_path = Path(dir) / new_file_name
        self.logger.info(f"Constructed file path: {file_path}")
        return file_path

    def create_read_file_path(self, file_ext: str, dir: str = None, split_name: str = None):
        file_path = self._create_file_path(file_ext=file_ext, dir=dir, split_name=split_name)
        if not file_path.exists():
            raise_error_logger(self.logger, f"File {file_path} does not exist.")
        return file_path

    def create_write_file_path(self, file_ext: str, dir: str = None, split_name: str = None):
        file_path = self._create_file_path(file_ext=file_ext, dir=dir, split_name=split_name)
        if file_path.exists():
            self.logger.warning(f"File {file_path} already exists and might be overwritten.")
        return file_path
