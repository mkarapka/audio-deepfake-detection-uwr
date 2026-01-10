from src.pipelines.best_balance_pipeline import BestBalancePipeline

if __name__ == "__main__":
    pipeline = BestBalancePipeline()
    pipeline.train_all_balancers()
    print("Training completed.")
    print(pipeline.pick_best_model())
