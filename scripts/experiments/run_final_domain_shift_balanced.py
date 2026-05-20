from scripts.experiments.run_experiment import ExperimentPreprocessConfig, RunExperiment
from src.common.experiment_configs import (
    BalanceType,
    ModelType,
    RunExperimentParams,
)


def get_expression(config_type: str) -> str:
    return f'anomaly == {1.0} or config.str.contains("{config_type}")'


if __name__ == "__main__":
    FEAT_SUFFIX = "_domain_shift"
    experiment_params = RunExperimentParams(
        description="Final train for $x0 on $x1 using best params from W&B artifacts (FFT vs WavLM)",
        experiment_group="final_domain_shift_comparison_balanced_test",
        epochs=20,
        fraction=1.0,
        batch_size=128,
        feat_suffix=FEAT_SUFFIX,
        params_artifact_type="model_params",
        models_types=[ModelType.LOGISTIC_REGRESSION, ModelType.MLP],
        balance_strategy={
            "dev": (BalanceType.UNDERSAMPLE.value, 1.0),
            "test": (BalanceType.UNDERSAMPLE.value, 1.0),
        },
        params_artifact_alias="latest",
        job_type="final_train",
        num_workers=-1,
        use_pos_weight=True,
    )
    real_vocoder_cfg = ExperimentPreprocessConfig(
        splits_names=["train", "dev", "test"],
        fraction=1.0,
        use_audio_id_sampling=False,
        use_standardize=True,
        balance_splits_strategy=None,
        remove_by_query={
            "train": get_expression("tts"),
            "dev": get_expression("tts"),
            "test": get_expression("vocoders"),
        },
    ).get_dict()
    real_tts_cfg = ExperimentPreprocessConfig(
        splits_names=["train", "dev", "test"],
        fraction=1.0,
        use_audio_id_sampling=False,
        use_standardize=True,
        balance_splits_strategy=None,
        remove_by_query={
            "train": get_expression("vocoders"),
            "dev": get_expression("vocoders"),
            "test": get_expression("tts"),
        },
    ).get_dict()

    preprocess_configs = {
        "fft_real_vocoders": real_vocoder_cfg,
        "fft_real_tts": real_tts_cfg,
    }
    run = RunExperiment(
        experiment_params=experiment_params,
        preprocess_configs=preprocess_configs,
        experiment_class="final_evaluation")
    run.run()
