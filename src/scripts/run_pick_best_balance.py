from src.pipelines.best_balance_pipeline import BestBalancePipeline

if __name__ == "__main__":
    pipeline = BestBalancePipeline()
    pipeline.train_on_raw_data(max_iter=200, n_trials=20)
    pipeline._train_clf_on_resampled_data("undersample")
    pipeline._train_clf_on_resampled_data("mix")
    pipeline._train_clf_on_resampled_data("oversample")
    print("Training completed.")
    print(pipeline.pick_best_model())
