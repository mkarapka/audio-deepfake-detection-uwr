import torch
import torch.nn as nn

from src.common.basic_functions import print_green
from src.models.logistic_regression_classifier import LogisticRegressionClassifier


class LogisticRegressionClassifierTest:
    def test_model_initialization(self):
        in_features = 128
        model = LogisticRegressionClassifier(in_features=in_features, device="cpu")

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
        model = LogisticRegressionClassifier(in_features=in_features, device="cpu")

        # Generowanie sztucznych danych - batch o rozmiarze 10 i odpowiedniej liczbie cech
        X = torch.randn(batch_size, in_features)

        # Puszczenie danych przez metodę forward u stworzonej klasy
        y_hat = model.forward(X)

        # Walidacja outputu
        assert y_hat.shape == (batch_size, 1), f"Nieoczekiwany shape od modelu: {y_hat.shape}"
        assert y_hat.requires_grad, "Tensor nie żąda dołączenia do gradient trackera. Złe rzutowanie."


if __name__ == "__main__":
    tester = LogisticRegressionClassifierTest()
    tester.test_model_initialization()
    tester.test_forward_pass()
    print_green("\n>>> LogisticRegressionClassifierTest: Wszystkie asercje przeszły bez błędu!")
