from scripts.experiments.run_experiment import ExperimentPreprocessConfig, RunExperiment
from src.common.experiment_configs import (
    BalanceStrategy,
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
    FEAT_SUFFIX = ""
    experiment_params = RunExperimentParams(
        description="Final train for $x0 on $x1 using best params from W&B artifacts (FFT vs WavLM)",
        experiment_group="fft_vs_wavlm_comparison",
        epochs=20,
        fraction=1.0,
        batch_size=128,
        feat_suffix=FEAT_SUFFIX,
        params_artifact_type="model_params",
        models_types=[ModelType.LOGISTIC_REGRESSION, ModelType.MLP],
        balance_strategy=None,
        params_artifact_alias="latest",
        job_type="final_train",
        num_workers=-1,
        use_pos_weight=True,
    )
    preprocess_configs = make_preprocess_configs(
        fraction=experiment_params.fraction,
        balance_strategy=experiment_params.balance_strategy,
        feat_suffix=experiment_params.feat_suffix,
    )

    run = RunExperiment(experiment_params=experiment_params, preprocess_configs=preprocess_configs)
    run.run()
