from src.common.constants import Constants as consts
from src.models.logistic_regression_classifier import LogisticRegressionClassifier
from src.preprocessing.feature_loader import FeatureLoader

if __name__ == "__main__":
    feature_loader = FeatureLoader(file_name=consts.feature_extracted)
    train_split = feature_loader.load_split_file(split_name="train")
    dev_split = feature_loader.load_split_file(split_name="dev")

    train_embeddings = feature_loader.load_embeddings_from_metadata(train_split)
    dev_embeddings = feature_loader.load_embeddings_from_metadata(dev_split)

    clf = LogisticRegressionClassifier(
        X_train=train_embeddings,
        y_train=train_split["target"],
        X_dev=dev_embeddings,
        y_dev=dev_split["target"],
    )
    clf.train()
    best_clf = clf.get_best_model()
    print("Training completed.")
    print(f"Best model coefficients: {best_clf.coef_}")
    print(f"Best model intercept: {best_clf.intercept_}")
