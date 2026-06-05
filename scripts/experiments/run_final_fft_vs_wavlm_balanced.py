from scripts.experiments.run_experiment import ExperimentPreprocessConfig, RunExperiment
from src.common.constants import Constants as consts
from src.common.experiment_configs import (
    BalanceStrategy,
    BalanceType,
    ModelType,
    RunExperimentParams,
)


def make_preprocess_configs(
    fraction: float,
    balance_strategy: BalanceStrategy | None,
    feat_suffix: str,
    splits_names: list[str] = ["train", "dev", "test"],
):
    wavlm = ExperimentPreprocessConfig(
        splits_names=splits_names,
        fraction=fraction,
        use_audio_id_sampling=False,
        use_standardize=False,
        balance_splits_strategy=balance_strategy,
        remove_by_query="anomaly == 1.0",
    ).get_dict()

    fft = wavlm.copy()
    fft["use_standardize"] = True

    return {f"fft{feat_suffix}": fft, f"wavlm{feat_suffix}": wavlm}


if __name__ == "__main__":
    EXP_SUFFIX = ""
    experiment_params = RunExperimentParams(
        description="Final train for $x0 on $x1 using best params from W&B artifacts (FFT vs WavLM)",
        experiment_group="fft_vs_wavlm_comparison_balanced_const_hid_sizes",
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
    preprocess_configs = make_preprocess_configs(
        fraction=experiment_params.fraction,
        balance_strategy=experiment_params.balance_strategy,
        experiment_suffix=experiment_params.experiment_suffix,
    )

    run = RunExperiment(
        experiment_params=experiment_params,
        preprocess_configs=preprocess_configs,
        experiment_class="final_evaluation",
    )
    run.run()
