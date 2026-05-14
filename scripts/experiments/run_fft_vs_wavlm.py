import wandb
from src.common.experiment_configs import (
    ExperimentConfig,
    ExperimentInfo,
    ExperimentPreprocessConfig,
    OptunaTrainingConfig,
    TorchParameters,
)
from src.common.wandb_config import WANDB_ENTITY, WANDB_PROJECT
from src.pipelines.experiments.fft_vs_wavlm import FFTvsWavLMExperiment
from src.training.objectives import LogisticRegressionObjective, MlpObjective

EXPERIMENT_NAME = "fft_vs_wavlm"
N_TRIALS = 1
EPOCHS = 2
FRACTION = 0.05
BATCH_SIZE = 32
NUM_WORKERS = -1
USE_POS_WEIGHT = True

if __name__ == "__main__":
    expr = "starting_point > 36.0"
    wavlm_preprocess_config = ExperimentPreprocessConfig(
        splits_names=["train", "dev"],
        fraction=FRACTION,
        use_audio_id_sampling=False,
        use_standardize=False,
        balance_splits_strategy=None,
        remove_by_query=expr,
    ).get_dict()
    fft_preprocess_config = wavlm_preprocess_config.copy()
    fft_preprocess_config["use_standardize"] = True

    torch_params = TorchParameters(
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        epochs=EPOCHS,
        use_pos_weight=USE_POS_WEIGHT,
    )
    optuna_training_config = OptunaTrainingConfig(
        objectives=[LogisticRegressionObjective, MlpObjective],
        n_trials=N_TRIALS,
        torch_params=torch_params,
    )

    experiment_config = ExperimentConfig(
        preprocess_configs={"fft": fft_preprocess_config, "wavlm": wavlm_preprocess_config},
        optuna_training_config=optuna_training_config,
    )

    experiment_info = ExperimentInfo(
        experiment_name=EXPERIMENT_NAME,
        models=["logistic_regression", "MLP"],
        description="Testing FFT vs WavLM features on Logistic Regression and MLP",
        config=experiment_config,
    )

    run = wandb.init(
        project=WANDB_PROJECT,
        entity=WANDB_ENTITY,
        name=experiment_info.experiment_name,
        config=experiment_info,
    )

    experiment = FFTvsWavLMExperiment(experiment_info=experiment_info, wandb_run=run)
    experiment.run()
