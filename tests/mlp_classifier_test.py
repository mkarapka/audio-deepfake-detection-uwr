import torch

from src.common.constants import Constants as consts
from src.common.utils import print_green
from src.models.mlp_classifier import MlpClassifier


class MlpClassifierTest:
    def test_model_initialization(self):
        input_size = 16
        hidden_sizes = [32, 16]
        dropout_rate = 0.2

        model = MlpClassifier(
            input_size=input_size,
            hidden_sizes=hidden_sizes,
            dropout_rate=dropout_rate,
            device="cpu",
        )

        assert model.model is not None, "Model nie został przekazany do self.model"
        assert model.input_size == input_size, "input_size powinien zostać zapamiętany w obiekcie"
        assert model.hidden_sizes == hidden_sizes, "hidden_sizes powinny zostać zapamiętane w obiekcie"
        assert model.dropout_rate == float(dropout_rate), "dropout_rate powinien zostać zapamiętany w obiekcie"

    def test_forward_pass(self):
        torch.manual_seed(27)
        input_size = 16
        hidden_sizes = [32, 16]
        dropout_rate = 0.0
        batch_size = 5

        model = MlpClassifier(
            input_size=input_size,
            hidden_sizes=hidden_sizes,
            dropout_rate=dropout_rate,
            device="cpu",
        )

        X = torch.randn(batch_size, input_size)
        y_hat = model.forward(X)

        assert y_hat.shape == (batch_size, 1), f"Nieoczekiwany shape od modelu: {y_hat.shape}"
        assert y_hat.requires_grad, "Tensor nie żąda dołączenia do gradient trackera."

    def test_save_and_from_pretrained_roundtrip(self):
        torch.manual_seed(27)
        input_size = 16
        hidden_sizes = [32, 16]
        dropout_rate = 0.0
        batch_size = 4

        model = MlpClassifier(
            input_size=input_size,
            hidden_sizes=hidden_sizes,
            dropout_rate=dropout_rate,
            device="cpu",
        )
        X = torch.randn(batch_size, input_size)
        y_before = model.forward(X).detach().clone()

        save_dir = consts.tests_data_dir / "models"
        save_dir.mkdir(parents=True, exist_ok=True)
        file_path = save_dir / "mlp_roundtrip_test.pt"
        model.save(file_path)

        loaded = MlpClassifier.from_pretrained(str(file_path), device="cpu")
        y_after = loaded.forward(X).detach().clone()

        assert file_path.exists(), "Expected saved model file to exist."
        assert loaded.input_size == input_size, "Expected input_size to be restored from checkpoint."
        assert loaded.hidden_sizes == hidden_sizes, "Expected hidden_sizes to be restored from checkpoint."
        assert loaded.dropout_rate == float(dropout_rate), "Expected dropout_rate to be restored from checkpoint."
        assert torch.allclose(y_before, y_after), "Expected loaded model to produce identical outputs."


if __name__ == "__main__":
    tester = MlpClassifierTest()
    tester.test_model_initialization()
    tester.test_forward_pass()
    tester.test_save_and_from_pretrained_roundtrip()
    print_green("\n>>> MlpClassifierTest: Wszystkie asercje przeszły bez błędu!")


def test_mlp_model_initialization():
    MlpClassifierTest().test_model_initialization()


def test_mlp_forward_pass():
    MlpClassifierTest().test_forward_pass()


def test_mlp_save_and_from_pretrained_roundtrip():
    MlpClassifierTest().test_save_and_from_pretrained_roundtrip()
