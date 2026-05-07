from pathlib import Path

import joblib

from src.common.constants import Constants as consts
from src.common.logger import raise_error_logger, setup_logger
from src.models.base_model import BaseModel


class ArtifactManager:
    def __init__(self, experiment_name: str):
        self.logger = setup_logger(__class__.__name__, log_to_console=True)
        self.experiment_name = experiment_name

    def _increase_file_number(self, file_path: Path) -> Path:
        if not file_path.exists():
            return file_path
        stem = file_path.stem
        suffix = file_path.suffix
        if "_" in stem and stem.split("_")[-1].isdigit():
            base_stem = "_".join(stem.split("_")[:-1])
            number = int(stem.split("_")[-1]) + 1
        else:
            base_stem = stem
            number = 0
        new_file_path = file_path.parent / f"{base_stem}_{number}{suffix}"
        return self._increase_file_number(new_file_path)

    def _generate_file_path(self, file_name: str, ext: str, main_dir: Path) -> Path:
        if not main_dir.exists():
            main_dir.mkdir(parents=True, exist_ok=True)

        final_path = main_dir / self.experiment_name / f"{file_name}.{ext}"
        if final_path.exists():
            final_path = self._increase_file_number(final_path)
            self.logger.warning(f"File {final_path} already exists. Saving to {final_path} instead.")

        return final_path

    def _get_file_path(self, file_name: str, ext: str, main_dir: Path) -> Path:
        file_path = main_dir / self.experiment_name / f"{file_name}.{ext}"
        if not file_path.exists():
            raise_error_logger(self.logger, f"File not found: {file_path}")
        return file_path

    def get_model_file_path(self, file_name: str, ext: str) -> Path:
        return self._get_file_path(file_name=file_name, ext=ext, main_dir=consts.models_dir)

    def get_params_file_path(self, file_name: str, ext: str) -> Path:
        return self._get_file_path(file_name=file_name, ext=ext, main_dir=consts.params_dir)

    def load_model(self, model: BaseModel, model_name: str, ext: str):
        file_path = self.get_model_file_path(file_name=model_name, ext=ext)
        self.logger.info(f"Loading model from {file_path}")
        model.load(file_path=file_path)
        self.logger.info("Model loaded successfully.")

    def load_params(self, file_name: str):
        file_path = self.get_params_file_path(file_name=file_name, ext="pkl")
        self.logger.info(f"Loading params from {file_path}")
        params = joblib.load(file_path)
        self.logger.info("Params loaded successfully.")
        return params

    def save_model(self, model: BaseModel, file_name: str, ext: str):
        file_path = self.get_model_file_path(file_name=file_name, ext=ext)
        self.logger.info(f"Saving model to {file_path}")
        model.save(file_path=file_path)
        self.logger.info("Model saved successfully.")

    def save_params(self, params: dict, file_name: str):
        file_path = self.get_params_file_path(file_name=file_name, ext="pkl")
        self.logger.info(f"Saving params to {file_path}")
        joblib.dump(params, file_path)
        self.logger.info("Params saved successfully.")
