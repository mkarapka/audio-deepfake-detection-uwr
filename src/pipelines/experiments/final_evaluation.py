import wandb
from src.common.constants import Constants as consts
from src.common.experiment_configs import ExperimentInfo
from src.common.logger import WandbLogger, raise_error_logger, setup_logger
from src.evaluation.binary_evaluator import BinaryEvaluator
from src.preprocessing.experiment_preprocessor import ExperimentPreprocessor
from src.training.artifact_manager import ArtifactManager


class FinalEvaluationExperiment:
    def __init__(self, *, experiment_info: ExperimentInfo, wandb_run: wandb.Run, feat_suffix: str = ""):
        self.experiment_config = experiment_info.config
        self.wandb_run = wandb_run
        self.feat_suffix = feat_suffix

        self.logger = setup_logger(__class__.__name__, log_to_console=True)
        self.wandb_logger = WandbLogger(self.logger, run=self.wandb_run)

        self.training_config = self.experiment_config.training_config
        self.torch_params = self.training_config.torch_params

        self.artifact_manager = ArtifactManager(experiment_name=experiment_info.experiment_name)

    def _get_feat_suffix(self, feature_key: str) -> str:
        if "wavlm" in feature_key:
            return consts.wavlm_emb_suffix
        if "fft" in feature_key:
            return consts.fft_emb_suffix
        return "unknown"

    def run(self):
        if self.torch_params is None:
            raise_error_logger(self.logger, "torch_params is required for final training")

        feature_type_dataloaders_map = {}
        for feature_key, preprocess_config in self.experiment_config.preprocess_configs.items():
            feat_suffix = self._get_feat_suffix(feature_key)
            preprocessor = ExperimentPreprocessor(feat_suffix=feat_suffix, load_file_name=preprocess_config.file_name)

            self.wandb_logger.info(f"Preprocessing {feature_key} features with config: {preprocess_config}...")
            dataset_map = preprocessor.preprocess_data(**preprocess_config)

            self.wandb_logger.info(f"Getting Dataloaders for {feature_key} features...")
            dataloaders_map = preprocessor.get_dataloaders(dataset_map, batch_size=self.torch_params.batch_size)
            self.wandb_logger.info(f"Dataloader keys for {feature_key} features: {dataloaders_map.keys()}")

            feature_type_dataloaders_map[feature_key] = dataloaders_map

        for feature_key, dataloaders_map in feature_type_dataloaders_map.items():
            test_loader = dataloaders_map.get("test")
            if test_loader is None:
                raise_error_logger(self.logger, f"Missing 'test' dataloader for feature_key={feature_key}")

            for model_type in self.training_config.models:
                model_name = model_type.value

                classifier = self.artifact_manager.load_model_from_wandb(
                    wandb_run=self.wandb_run,
                    artifact_name=f"{model_name}_{feature_key}_final_model",
                    artifact_type=f"model{self.feat_suffix}",
                    alias="latest",
                )

                log_prefix = f"final_evaluation/{model_name}/{feature_key}"

                self.wandb_logger.info(f"Running final test evaluation for {model_name} / {feature_key}...")
                evaluator = BinaryEvaluator()
                test_metrics = evaluator.evaluate(
                    model=classifier,
                    dataloader=test_loader,
                    log_prefix=f"{log_prefix}/test",
                )
                self.wandb_logger.log_metrics(test_metrics)
                self.wandb_logger.info(f"Test metrics for {model_name} / {feature_key}: {test_metrics}")

        self.wandb_run.finish()
