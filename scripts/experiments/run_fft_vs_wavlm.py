import wandb
from src.common.experiment_configs import (
    BalanceType,
    ExperimentConfig,
    ExperimentInfo,
    ExperimentPreprocessConfig,
    ModelType,
    OptunaTrainingConfig,
    TorchParameters,
)
from src.common.wandb_config import WANDB_ENTITY, WANDB_PROJECT
from src.pipelines.experiments.fft_vs_wavlm import FFTvsWavLMExperiment
from src.training.objectives import (
    LogisticRegressionObjective,
    MlpObjective,
    XGBoostObjective,
)

EXPERIMENT_NAME = "fft_vs_wavlm"
EXPERIMENT_GROUP = "fft_vs_wavlm_search_hyperparams"
DESCRIPTION = "Optuna hyperparameter search for $x0 on $x1 using Logistic Regression, MLP and XGBoost classifiers to compare FFT vs WavLM features"

N_TRIALS = 30
EPOCHS = 20
FRACTION_DICT = {"train": 0.3, "dev": 1.0}
BALANCE_STRATEGY = {"dev": (BalanceType.UNDERSAMPLE.value, 1.0)}
BATCH_SIZE = 128
NUM_WORKERS = -1
USE_POS_WEIGHT = True


def make_preprocess_configs():
    wavlm = ExperimentPreprocessConfig(
        splits_names=["train", "dev"],
        fraction=FRACTION_DICT,
        use_audio_id_sampling=False,
        use_standardize=False,
        balance_splits_strategy=BALANCE_STRATEGY,
        remove_by_query="anomaly == 1.0",
    ).get_dict()

    fft = wavlm.copy()
    fft["use_standardize"] = True

    return {"wavlm": wavlm, "fft": fft}


def get_model_name(objective_cls):
    if objective_cls is LogisticRegressionObjective:
        return ModelType.LOGISTIC_REGRESSION.value
    elif objective_cls is MlpObjective:
        return ModelType.MLP.value
    elif objective_cls is XGBoostObjective:
        return ModelType.XGBOOST.value
    else:
        raise ValueError(f"Unsupported objective class: {objective_cls.__name__}")


if __name__ == "__main__":
    preprocess_configs = make_preprocess_configs()

    torch_params = TorchParameters(
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        epochs=EPOCHS,
        use_pos_weight=USE_POS_WEIGHT,
    )

    objectives = [LogisticRegressionObjective, MlpObjective]

    for feature_key, preprocess_cfg in preprocess_configs.items():
        for objective in objectives:

            optuna_training_config = OptunaTrainingConfig(
                objectives=[objective],
                n_trials=N_TRIALS,
                torch_params=torch_params,
            )

            experiment_config = ExperimentConfig(
                preprocess_configs={feature_key: preprocess_cfg},
                training_config=optuna_training_config,
            )

            model_name = get_model_name(objective)
            run_name = f"{EXPERIMENT_GROUP}/{model_name}/{feature_key}"

            experiment_info = ExperimentInfo(
                experiment_name=run_name,
                models=[model_name],
                description=DESCRIPTION.replace("$x0", f"{feature_key}").replace("$x1", f"{model_name}"),
                config=experiment_config,
            )

            run = wandb.init(
                project=WANDB_PROJECT,
                entity=WANDB_ENTITY,
                group=EXPERIMENT_GROUP,
                name=experiment_info.experiment_name,
                config=experiment_info,
            )

            experiment = FFTvsWavLMExperiment(experiment_info=experiment_info, wandb_run=run)
            experiment.run()
