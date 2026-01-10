from src.pipelines.best_balance_pipeline import BestBalancePipeline

if __name__ == "__main__":
    pipeline = BestBalancePipeline()
    pipeline._train_clf_on_resampled_data("undersample")
    pipeline._train_clf_on_resampled_data("mix")
    print("Training completed.")
    print(pipeline.pick_best_model())
