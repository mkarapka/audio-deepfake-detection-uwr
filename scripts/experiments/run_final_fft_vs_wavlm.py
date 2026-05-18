import wandb
from src.common.experiment_configs import (
    ExperimentConfig,
    ExperimentInfo,
    ExperimentPreprocessConfig,
    FinalTrainConfig,
    ModelType,
    TorchParameters,
)
from src.common.wandb_config import WANDB_ENTITY, WANDB_PROJECT
from src.pipelines.experiments.final_train_experiment import FinalTrainExperiment

EXPERIMENT_NAME = "fft_vs_wavlm_final_train_test"
EPOCHS = 15
FRACTION = 0.1
FRACTION_DICT = {"train": 0.05, "dev": 1.0, "test": 1.0}
BATCH_SIZE = 32
NUM_WORKERS = -1
USE_POS_WEIGHT = True

DESCRIPTION = "Final train for $x0 on $x1 using best params from W&B artifacts (FFT vs WavLM) on Logistic Regression and MLP classifiers"

EXPERIMENT_GROUP = "fft_vs_wavlm_comparison"
PARAMS_ARTIFACT_ALIAS = "latest"  # change to e.g. "v0" if you want a fixed version
PARAMS_ARTIFACT_TYPE = "model_params"  # must match type used in hyperparam search run
# PARAMS_ARTIFACT_TYPE = "test_params"  # must match type used in hyperparam search run


def make_preprocess_configs():
    wavlm = ExperimentPreprocessConfig(
        splits_names=["train", "dev", "test"],
        fraction=FRACTION_DICT,
        use_audio_id_sampling=False,
        use_standardize=False,
        balance_splits_strategy=None,
        remove_by_query="anomaly == 1.0",
    ).get_dict()

    fft = wavlm.copy()
    fft["use_standardize"] = True

    return {"fft": fft}


if __name__ == "__main__":
    torch_params = TorchParameters(
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        epochs=EPOCHS,
        use_pos_weight=USE_POS_WEIGHT,
    )

    preprocess_configs = make_preprocess_configs()
    models = [ModelType.LOGISTIC_REGRESSION]

    for feature_key, preprocess_cfg in preprocess_configs.items():
        for model in models:
            training_config = FinalTrainConfig(
                models=[model],
                torch_params=torch_params,
                best_params_artifact_alias=PARAMS_ARTIFACT_ALIAS,
                best_params_artifact_type=PARAMS_ARTIFACT_TYPE,
            )
            experiment_config = ExperimentConfig(
                preprocess_configs={feature_key: preprocess_cfg},
                training_config=training_config,
            )

            run_name = f"{EXPERIMENT_GROUP}/{model.value}/{feature_key}"

            experiment_info = ExperimentInfo(
                experiment_name=run_name,
                models=training_config.models,
                description=DESCRIPTION.replace("$x0", f"{feature_key}").replace("$x1", f"{model.value}"),
                config=experiment_config,
            )
            run = wandb.init(
                project=WANDB_PROJECT,
                entity=WANDB_ENTITY,
                group=EXPERIMENT_GROUP,
                name=run_name,
                job_type="final_train",
                config=experiment_info,
            )

            experiment = FinalTrainExperiment(
                experiment_info=experiment_info,
                wandb_run=run,
            )
            experiment.run()
