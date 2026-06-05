import wandb
from src.common.experiment_configs import (
    ExperimentConfig,
    ExperimentInfo,
    ExperimentPreprocessConfig,
    FinalTrainConfig,
    RunExperimentParams,
    TorchParameters,
)
from src.common.logger import raise_error_logger, setup_logger
from src.common.wandb_config import WANDB_ENTITY, WANDB_PROJECT
from src.pipelines.experiments.final_evaluation import FinalEvaluationExperiment
from src.pipelines.experiments.final_train_experiment import FinalTrainExperiment


class RunExperiment:
    def __init__(
        self,
        experiment_params: RunExperimentParams,
        preprocess_configs: dict[str, ExperimentPreprocessConfig],
        experiment_class="final_train",
    ):
        self.logger = setup_logger(__class__.__name__, log_to_console=True)
        self.experiment_params = experiment_params
        self.preprocess_configs = preprocess_configs
        self.experiment_class = experiment_class

    def _get_artifact_type_with_suffix(self, feat_suffix: str) -> str:
        if feat_suffix == "":
            return self.experiment_params.params_artifact_type
        if feat_suffix[0] != "_":
            raise_error_logger(self.logger, f"feat_suffix should start with '_' if not empty, got: '{feat_suffix}'")
        return f"{self.experiment_params.params_artifact_type}{feat_suffix}"

    def run(self):
        torch_params = TorchParameters(
            batch_size=self.experiment_params.batch_size,
            num_workers=self.experiment_params.num_workers,
            epochs=self.experiment_params.epochs,
            use_pos_weight=self.experiment_params.use_pos_weight,
            early_stopping_patience=self.experiment_params.early_stopping_patience,
            early_stopping_min_delta=self.experiment_params.early_stopping_min_delta,
        )

        for feature_key, preprocess_cfg in self.preprocess_configs.items():
            for model in self.experiment_params.models_types:
                training_config = FinalTrainConfig(
                    models=[model],
                    torch_params=torch_params,
                    best_params_artifact_alias=self.experiment_params.params_artifact_alias,
                    best_params_artifact_type=self._get_artifact_type_with_suffix(
                        self.experiment_params.experiment_suffix
                    ),
                )
                experiment_config = ExperimentConfig(
                    preprocess_configs={feature_key: preprocess_cfg},
                    training_config=training_config,
                )

                run_name = f"{self.experiment_params.experiment_group}/{model.value}/{feature_key}"

                experiment_info = ExperimentInfo(
                    experiment_name=run_name,
                    models=training_config.models,
                    description=self.experiment_params.description.replace("$x0", f"{feature_key}").replace(
                        "$x1", f"{model.value}"
                    ),
                    config=experiment_config,
                )
                run = wandb.init(
                    project=WANDB_PROJECT,
                    entity=WANDB_ENTITY,
                    group=self.experiment_params.experiment_group,
                    name=run_name,
                    job_type=self.experiment_params.job_type,
                    config=experiment_info,
                )

                if self.experiment_class == "final_train":
                    experiment = FinalTrainExperiment(
                        experiment_info=experiment_info,
                        wandb_run=run,
                        experiment_suffix=self.experiment_params.experiment_suffix,
                        load_file_name=self.experiment_params.load_file_name,
                    )
                elif self.experiment_class == "final_evaluation":
                    experiment = FinalEvaluationExperiment(
                        experiment_info=experiment_info,
                        wandb_run=run,
                        experiment_suffix=self.experiment_params.experiment_suffix,
                        load_file_name=self.experiment_params.load_file_name,
                    )
                else:
                    raise_error_logger(self.logger, f"Unsupported experiment_class: {self.experiment_class}")

                experiment.run()
