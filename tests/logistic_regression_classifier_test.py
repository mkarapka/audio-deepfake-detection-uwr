import torch
import torch.nn as nn

from src.common.constants import Constants as consts
from src.common.utils import print_green
from src.models.logistic_regression_classifier import LogisticRegressionClassifier


class LogisticRegressionClassifierTest:
    def test_model_initialization(self):
        in_features = 128
        model = LogisticRegressionClassifier(input_size=in_features, device="cpu")

        # Sprawdzamy czy model zainicjalizował się w atrybucie BaseModel/TorchModel
        assert model.model is not None, "Model nie został przekazany do self.model"

        # Sprawdzamy czy nn.Sequential posiada właściwą pierwszą warstwę
        assert isinstance(model.model[0], nn.Linear), "Warstwa modelu to nie nn.Linear!"
        assert model.model[0].in_features == in_features, "Błyskawiczny problem z zainicjalizowanymi features"
        assert (
            model.model[0].out_features == 1
        ), "LogisticRegression powinien zwracać tylko jedno wyjście out_features=1"

    def test_forward_pass(self):
        in_features = 128
        batch_size = 10
        model = LogisticRegressionClassifier(input_size=in_features, device="cpu")

        # Generowanie sztucznych danych - batch o rozmiarze 10 i odpowiedniej liczbie cech
        X = torch.randn(batch_size, in_features)

        # Puszczenie danych przez metodę forward u stworzonej klasy
        y_hat = model.forward(X)

        # Walidacja outputu
        assert y_hat.shape == (batch_size, 1), f"Nieoczekiwany shape od modelu: {y_hat.shape}"
        assert y_hat.requires_grad, "Tensor nie żąda dołączenia do gradient trackera. Złe rzutowanie."

    def test_save_and_from_pretrained_roundtrip(self):
        torch.manual_seed(27)
        in_features = 32
        batch_size = 4

        model = LogisticRegressionClassifier(input_size=in_features, device="cpu")
        X = torch.randn(batch_size, in_features)
        y_before = model.forward(X).detach().clone()

        save_dir = consts.tests_data_dir / "models"
        save_dir.mkdir(parents=True, exist_ok=True)
        file_path = save_dir / "logreg_roundtrip_test.pt"
        model.save(file_path)

        loaded = LogisticRegressionClassifier.from_pretrained(str(file_path), device="cpu")
        y_after = loaded.forward(X).detach().clone()

        assert file_path.exists(), "Expected saved model file to exist."
        assert loaded.model[0].in_features == in_features, "Expected in_features to be restored from checkpoint."
        assert torch.allclose(y_before, y_after), "Expected loaded model to produce identical outputs."


if __name__ == "__main__":
    tester = LogisticRegressionClassifierTest()
    tester.test_model_initialization()
    tester.test_forward_pass()
    tester.test_save_and_from_pretrained_roundtrip()
    print_green("\n>>> LogisticRegressionClassifierTest: All tests passed!")


def test_logreg_model_initialization():
    LogisticRegressionClassifierTest().test_model_initialization()


def test_logreg_forward_pass():
    LogisticRegressionClassifierTest().test_forward_pass()


def test_logreg_save_and_from_pretrained_roundtrip():
    LogisticRegressionClassifierTest().test_save_and_from_pretrained_roundtrip()
