import wandb
from src.common.constants import Constants as consts
from src.common.experiment_configs import ExperimentInfo
from src.common.logger import WandbLogger, setup_logger
from src.preprocessing.experiment_preprocessor import ExperimentPreprocessor
from src.training.artifact_manager import ArtifactManager
from src.training.model_trainer import ModelTrainer


class FFTvsWavLMExperiment:
    def __init__(self, experiment_info: ExperimentInfo, wandb_run: wandb.Run):
        self.experiment_config = experiment_info.config
        self.wandb_run = wandb_run

        self.logger = setup_logger(__class__.__name__, log_to_console=True)
        self.wandb_logger = WandbLogger(self.logger, run=self.wandb_run)

        self.optuna_training_config = self.experiment_config.training_config
        self.torch_params = self.optuna_training_config.torch_params

        self.model_trainer = ModelTrainer()
        self.artifact_manager = ArtifactManager(experiment_name=experiment_info.experiment_name)

    def run(self):
        feature_type_dataloaders_map = {}
        for feature_key, preprocess_config in self.experiment_config.preprocess_configs.items():
            feat_suffix = (
                consts.wavlm_emb_suffix
                if "wavlm" in feature_key
                else consts.fft_emb_suffix if "fft" in feature_key else "unknown"
            )
            preprocessor = ExperimentPreprocessor(feat_suffix=feat_suffix)

            self.wandb_logger.info(f"Preprocessing {feature_key} features with config: {preprocess_config}...")
            dataset_map = preprocessor.preprocess_data(**preprocess_config)

            self.wandb_logger.info(f"Getting Dataloaders for {feature_key} features...")
            dataloaders_map = preprocessor.get_dataloaders(dataset_map, batch_size=self.torch_params.batch_size)
            self.wandb_logger.info(f"Dataloader keys for {feature_key} features: {dataloaders_map.keys()}")

            feature_type_dataloaders_map[feature_key] = dataloaders_map

        for feature_key, dataloaders_map in feature_type_dataloaders_map.items():
            train_dataloader = dataloaders_map["train"]
            dev_dataloader = dataloaders_map["dev"]
            for objective_cls in self.optuna_training_config.objectives:
                obj_name = objective_cls.__name__
                self.wandb_logger.info(f"Training {obj_name} with {feature_key} features...")

                best_params, best_value = self.model_trainer.optuna_train(
                    objective=objective_cls(
                        train_loader=train_dataloader,
                        val_loader=dev_dataloader,
                        wandb_run=self.wandb_run,
                    ),
                    n_trials=self.optuna_training_config.n_trials,
                    epochs=self.torch_params.epochs,
                )

                self.wandb_logger.info(f"Best params for {obj_name} with {feature_key} features: {best_params}")
                self.wandb_logger.log_metrics({f"{obj_name}_{feature_key}_best_value": best_value})

                file_name = f"{obj_name}_{feature_key}_best_params"
                self.artifact_manager.save_params(
                    params=best_params,
                    file_name=file_name,
                )
                file_path = self.artifact_manager.get_params_file_path(file_name=file_name)
                self.wandb_run.log_artifact(file_path, name=file_name, type="model_params")

                self.wandb_logger.info(
                    f"Saved best params for {obj_name} with {feature_key} features to {file_path} and logged to wandb."
                )
        self.wandb_run.finish()
