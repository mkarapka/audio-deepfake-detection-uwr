import wandb
from src.common.experiment_configs import ExperimentInfo, ExperimentPreprocessConfig
from src.common.wandb_config import WANDB_ENTITY, WANDB_PROJECT
from src.pipelines.experiments.fft_vs_wavlm import FFTvsWavLMExperiment

N_TRIALS = 1
EPOCHS = 2
USE_POS_WEIGHT = True
FRACTION = 0.05

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

    experiment_info = ExperimentInfo(
        experiment_name="fft_vs_wavlm_7",
        models=["logistic_regression", "MLP"],
        description="Testing FFT vs WavLM features on Logistic Regression and MLP",
        experiment_preprocess_configs={"fft": fft_preprocess_config, "wavlm": wavlm_preprocess_config},
        n_trials=N_TRIALS,
        objective_params={"epochs": EPOCHS, "use_pos_weight": USE_POS_WEIGHT},
    )

    run = wandb.init(
        project=WANDB_PROJECT,
        entity=WANDB_ENTITY,
        name=experiment_info.experiment_name,
        config=experiment_info,
    )
    experiment = FFTvsWavLMExperiment(experiment_info=experiment_info, wandb_run=run)
    experiment.run()
