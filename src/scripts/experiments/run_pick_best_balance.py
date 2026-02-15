from src.models.mlp_classifier import MLPClassifier
from src.pipelines.experiments.best_balance_pipeline import BestBalancePipeline

if __name__ == "__main__":
    pipeline = BestBalancePipeline(
        clf_model=MLPClassifier,
    )
    pipeline.train_all_balancers()
    pipeline.create_dataframe_and_save(file_name="best_balance_results")
    print("Training completed.")
    print(pipeline.pick_best_model())
