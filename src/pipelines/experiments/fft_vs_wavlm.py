import wandb

from src.common.constants import Constants as consts, ExperimentInfo
from src.common.logger import setup_logger
from src.preprocessing.experiment_preprocessor import ExperimentPreprocessor
from src.training.artifact_manager import ArtifactManager
from src.training.model_trainer import ModelTrainer
from src.training.objectives import LogisticRegressionObjective, MlpObjective


class FFTvsWavLMExperiment:
    def __init__(self, experiment_info: dict[str, ExperimentInfo], wandb_run=None):
        self.logger = setup_logger(__class__.__name__, log_to_console=True, wandb_run=wandb_run)
        self.wandb_run = wandb_run
        self.n_trials = experiment_info.n_trials
        self.experiment_info = experiment_info
        self.fft_preprocess_config = self.experiment_info.experiment_preprocess_configs["fft"]
        self.wavlm_preprocess_config = self.experiment_info.experiment_preprocess_configs["wavlm"]
        self.model_trainer = ModelTrainer()
        self.artifact_manager = ArtifactManager(experiment_name=self.experiment_info.experiment_name)

    def run(self):
        self.logger.info(self.wavlm_preprocess_config)
        wavlm_preprocessor = ExperimentPreprocessor(feat_suffix=consts.wavlm_emb_suffix)
        self.logger.info("Preprocessing WavLM features...")
        wavlm_dataset_map = wavlm_preprocessor.preprocess_data(**self.wavlm_preprocess_config)

        self.logger.info("Getting Dataloader for training...")
        wavlm_dataloaders_map = wavlm_preprocessor.get_dataloaders(wavlm_dataset_map, batch_size=32)
        self.logger.info(f"Dataloader keys: {wavlm_dataloaders_map.keys()}")

        self.logger.info(self.fft_preprocess_config)
        fft_preprocessor = ExperimentPreprocessor(feat_suffix=consts.fft_emb_suffix)
        self.logger.info("Preprocessing fft features...")
        fft_dataset_map = fft_preprocessor.preprocess_data(**self.fft_preprocess_config)

        self.logger.info("Getting Dataloader for training...")
        fft_dataloaders_map = fft_preprocessor.get_dataloaders(fft_dataset_map, batch_size=32)
        self.logger.info(f"Dataloader keys: {fft_dataloaders_map.keys()}")

        dataloaders_dict = {
            "wavlm": wavlm_dataloaders_map,
            "fft": fft_dataloaders_map,
        }

        model_trainer = ModelTrainer()
        artifact_manager = ArtifactManager(experiment_name=self.experiment_info.experiment_name)

        objectives = [LogisticRegressionObjective, MlpObjective]
        objectives_params = self.experiment_info.objective_params

        for feature_key, dataloaders_map in dataloaders_dict.items():
            train_dataloader = dataloaders_map["train"]
            dev_dataloader = dataloaders_map["dev"]
            for objective_cls in objectives:
                self.logger.info(f"Training {objective_cls.__name__} with {feature_key} features...")
                objective = objective_cls(train_loader=train_dataloader, val_loader=dev_dataloader)
                best_params, best_value = model_trainer.optuna_train(
                    objective=objective, n_trials=self.n_trials, **objectives_params
                )
                self.logger.info(
                    f"Best params for {objective_cls.__name__} with {feature_key} features: {best_params}, best value: {best_value}"
                )
                file_name = f"{objective_cls.__name__}_{feature_key}_best_params"
                artifact_manager.save_params(
                    params=best_params,
                    file_name=file_name,
                )
                params_file_path = artifact_manager.get_params_file_path(file_name=file_name)
                wandb.save(params_file_path)
                self.logger.info(
                    f"Saved best params for {objective_cls.__name__} with {feature_key} features to {params_file_path} and logged to wandb."
                )
        self.wandb_run.finish()
