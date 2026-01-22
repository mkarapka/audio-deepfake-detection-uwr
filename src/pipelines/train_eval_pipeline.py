from src.common.constants import Constants as consts
from src.common.logger import setup_logger
from src.models.mlp_classifier import MLPClassifier
from src.preprocessing.collector import Collector
from src.preprocessing.feature_loader import FeatureLoader


class TrainEvalPipeline:
    def __init__(self):
        self.logger = setup_logger(__class__.__name__, log_to_console=True)
        self.feature_loader = FeatureLoader(file_name=consts.feature_extracted)
        self.collector = Collector(save_file_name="train_eval_results.csv")
        self.classifier = MLPClassifier(is_chunk_prediction=False)

    def run(self):
        self.logger.info("Loading training and evaluation data.")
        train_meta, train_embeddings = self.feature_loader.transform(split_name="train")
        sampled_train_meta, sampled_train_embeddings = self.feature_loader.sample_fraction_metadata_and_embeddings(
            train_meta, train_embeddings, fraction=0.4
        )
        dev_meta, dev_embeddings = self.feature_loader.transform(split_name="dev")

        self.logger.info("Starting model training.")
        self.classifier.optuna_fit(
            n_trials=50,
            X_train=sampled_train_embeddings,
            y_train=sampled_train_meta["target"].values,
            X_dev=dev_embeddings,
            y_dev=dev_meta["target"].values,
        )


if __name__ == "__main__":
    pipeline = TrainEvalPipeline()
    pipeline.run()
