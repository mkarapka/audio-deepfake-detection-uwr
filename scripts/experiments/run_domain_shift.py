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

EXPERIMENT_NAME = "domain_shift"
N_TRIALS = 30
EPOCHS = 10
FRACTION = 0.3
BATCH_SIZE = 128
NUM_WORKERS = -1
USE_POS_WEIGHT = True

def get_expression(config_type: str) -> str:
    return f'anomaly == {1.0} and config.str.contains("{config_type}")'

if __name__ == "__main__":
    fft_real_tts_preprocess_config = ExperimentPreprocessConfig(
        splits_names=["train", "dev"],
        fraction=FRACTION,
        use_audio_id_sampling=False,
        use_standardize=True,
        balance_splits_strategy=None,
        remove_by_query=get_expression("vocoders"),
    ).get_dict()

    fft_real_vocoders_preprocess_config = ExperimentPreprocessConfig(
        splits_names=["train", "dev"],
        fraction=FRACTION,
        use_audio_id_sampling=False,
        use_standardize=True,
        balance_splits_strategy=None,
        remove_by_query=get_expression("tts"),
    ).get_dict()

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
        preprocess_configs={
            "fft_real_tts": fft_real_tts_preprocess_config,
            "fft_real_vocoders": fft_real_vocoders_preprocess_config,
        },
        training_config=optuna_training_config,
    )

    experiment_info = ExperimentInfo(
        experiment_name=EXPERIMENT_NAME,
        models=["LogisticRegression", "MLP"],
        description="Testing Domain Shift by training MLP on one of subclasses of data and evaluating on others.",
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
