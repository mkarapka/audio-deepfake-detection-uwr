from src.common.constants import Constants as consts
from src.pipelines.best_balance_pipeline import BestBalancePipeline

if __name__ == "__main__":
    pipeline = BestBalancePipeline(
        RATIOS_CONFIG=consts.only_mix_equal_ratio_config,
        objective="f1",
        is_chunk_prediction=True,
    )
    logger = pipeline.logger

    pipeline.train_all_balancers(reduce_factor=0.5, n_trials=50)
    pipeline.create_dataframe_and_save(file_name="best_balance_results")

    logger.info("Training completed.")
    logger.info(pipeline.pick_best_model())
