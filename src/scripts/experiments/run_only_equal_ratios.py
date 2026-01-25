from src.common.constants import Constants as consts
from src.pipelines.best_balance_pipeline import BestBalancePipeline
from src.models.mlp_classifier import MLPClassifier

if __name__ == "__main__":
    pipeline = BestBalancePipeline(
        clf_model=MLPClassifier,
        RATIOS_CONFIG=consts.only_equal_ratios_config, objective="f1", is_chunk_prediction=True
    )
    pipeline.train_all_balancers(reduce_factor=0.5, n_trials=20)
    pipeline.create_dataframe_and_save(file_name="best_balance_results")
    print("Training completed.")
    print(pipeline.pick_best_model())
