from scripts.experiments.run_experiment import ExperimentPreprocessConfig, RunExperiment
from src.common.constants import Constants as consts
from src.common.experiment_configs import (
    BalanceStrategy,
    BalanceType,
    ModelType,
    RunExperimentParams,
)


def get_expression(config_type: str) -> str:
    return f'anomaly == {1.0} or config.str.contains("{config_type}")'


def make_preprocess_config(
    fraction: float,
    spoofing_type: str,
    balance_strategy: BalanceStrategy | None,
    splits_names: list[str] = ["train", "dev", "test"],
):
    if spoofing_type == "tts":
        remove_dict = {
            "train": get_expression("vocoders"),
            "dev": get_expression("vocoders"),
            "test": get_expression("tts"),
        }
    else:
        remove_dict = {
            "train": get_expression("tts"),
            "dev": get_expression("tts"),
            "test": get_expression("vocoders"),
        }

    fft_cfg = ExperimentPreprocessConfig(
        splits_names=splits_names,
        fraction=fraction,
        use_audio_id_sampling=False,
        use_standardize=True,
        balance_splits_strategy=balance_strategy,
        remove_by_query=remove_dict,
    ).get_dict()

    return fft_cfg


if __name__ == "__main__":
    EXP_SUFFIX = "_domain_shift"
    experiment_params = RunExperimentParams(
        description="Final train for $x0 on $x1 using best params from W&B artifacts (FFT vs WavLM)",
        experiment_group="final_domain_shift_comparison_balanced_const_hid_sizes",
        epochs=consts.default_epochs,
        fraction=1.0,
        batch_size=consts.default_batch_size,
        experiment_suffix=EXP_SUFFIX,
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
        early_stopping_patience=consts.default_early_stopping_patience,
    )

    preprocess_configs = {
        "fft_real_vocoders": make_preprocess_config(
            fraction=experiment_params.fraction,
            spoofing_type="vocoders",
            balance_strategy=experiment_params.balance_strategy,
        ),
        "fft_real_tts": make_preprocess_config(
            fraction=experiment_params.fraction,
            spoofing_type="tts",
            balance_strategy=experiment_params.balance_strategy,
        ),
    }

    run = RunExperiment(
        experiment_params=experiment_params,
        preprocess_configs=preprocess_configs,
        experiment_class="final_evaluation",
    )
    run.run()
