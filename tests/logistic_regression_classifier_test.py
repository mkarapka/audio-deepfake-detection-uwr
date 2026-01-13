from sklearn.metrics import accuracy_score

from src.preprocessing.feature_loader import FeatureLoader
from training.logistic_regression_classifier import LogisticRegressionClassifier


class LogisticRegressionClassifierTest:
    def __init__(self, feature_loader: FeatureLoader):
        self.feature_loader = feature_loader
        self.train_split = self.feature_loader.load_split_file(split_name="train")
        self.dev_split = self.feature_loader.load_split_file(split_name="dev")

    def test_logistic_regression_trainer(self):
        print("Testing Logistic Regression Trainer...")
        print(f"Train split size: {self.train_split.shape}")
        print(f"Dev split size: {self.dev_split.shape}")
        print("Index types:", self.train_split.index.dtype, self.dev_split.index.dtype)
        print("Index samples:", self.train_split.index[:5], self.dev_split.index[:5])
        embeddings = self.feature_loader.load_embeddings_from_metadata(self.train_split)
        dev_embeddings = self.feature_loader.load_embeddings_from_metadata(
            self.dev_split
        )

        clf = LogisticRegressionClassifier(
            X_train=embeddings,
            y_train=self.train_split["target"],
            X_dev=dev_embeddings,
            y_dev=self.dev_split["target"],
        )
        clf.train()
        best_clf = clf.get_best_model()

        dev_predictions = best_clf.predict(dev_embeddings)
        accuracy = accuracy_score(self.dev_split["target"], dev_predictions)
        print(f"Logistic Regression Classifier Accuracy on Dev Set: {accuracy:.4f}")


if __name__ == "__main__":
    feature_loader = FeatureLoader(file_name="feature_extracted")
    test = LogisticRegressionClassifierTest(feature_loader=feature_loader)
    test.test_logistic_regression_trainer()
