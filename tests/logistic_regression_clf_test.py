import optuna

from models.logistic_regression_clf import LogisticRegressionClf, objective_acc


class LogisticRegressionClassifierTest:
    def test_model_initialization(self):
        print("Testing Logistic Regression Classifier Initialization...")
        clf = LogisticRegressionClf(C=1.0, class_weight="balanced", max_iter=100, random_state=42)
        expected_params = {
            "C": 1.0,
            "class_weight": "balanced",
            "max_iter": 100,
            "random_state": 42,
            "solver": "saga",
            "n_jobs": -1,
        }
        actual_params = clf.get_params()
        print("Actual Parameters:", actual_params)
        for param, expected_value in expected_params.items():
            actual_value = actual_params.get(param)
            assert actual_value == expected_value, f"Expected {param}={expected_value}, got {actual_value}"

        print("Logistic Regression Classifier Initialization Test Passed!")

    def test_objective_function(self):
        print("Testing Objective Function for Logistic Regression...")
        try:
            study = optuna.create_study(direction="maximize")
            model = LogisticRegressionClf
            features = {
                "X_train": [[0], [1]],
                "y_train": [0, 1],
                "X_dev": [[0], [1]],
                "y_dev": [0, 1],
            }
            study.optimize(lambda trial: objective_acc(trial, model=model, max_iter=10, **features), n_trials=5)
            print("Objective Function Test Passed with Accuracy:", study.best_value)
        except Exception as e:
            assert False, f"Objective function raised an exception: {e}"


if __name__ == "__main__":
    test = LogisticRegressionClassifierTest()
    test.test_model_initialization()
    test.test_objective_function()
    print("All Logistic Regression Classifier tests passed successfully!")
