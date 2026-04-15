import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.common.basic_functions import print_green
from src.models.torch_model import TorchModel


class TestableTorchModel(TorchModel):
    def load(self, model_name: str, ext: str, sub_dir: str = None):
        return None

    def save(self, model_name: str, ext: str, sub_dir: str = None):
        return None


class TorchModelTest:
    def _init_test_objects(self):
        torch.manual_seed(27)

        network = nn.Sequential(nn.Linear(2, 2))
        model = TestableTorchModel(model=network, class_name="TorchModelTest", include_mps=False)

        X = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [-1.0, -1.0]], dtype=torch.float32)
        y = torch.tensor([0, 1, 1, 0], dtype=torch.long)

        dataset = TensorDataset(X, y)
        loader = DataLoader(dataset, batch_size=2, shuffle=False)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        device = torch.device("cpu")

        return model, loader, criterion, optimizer, device

    def test_parameters(self):
        print("Testing TorchModel.parameters...")
        model, _, _, _, _ = self._init_test_objects()

        model_params = list(model.parameters())
        raw_model_params = list(model.model.parameters())

        assert len(model_params) > 0, "Expected model to expose at least one parameter."
        assert len(model_params) == len(raw_model_params), "Parameters count mismatch."
        for left, right in zip(model_params, raw_model_params):
            assert left is right, "TorchModel.parameters should forward underlying nn.Module parameters."

        print("TorchModel.parameters test passed.")

    def test_train_one_epoch(self):
        print("Testing TorchModel.train_one_epoch...")
        model, loader, criterion, optimizer, device = self._init_test_objects()

        params_before = [param.detach().clone() for param in model.model.parameters()]
        train_loss, train_acc = model.train_one_epoch(loader, criterion, optimizer, device)
        params_after = list(model.model.parameters())

        assert isinstance(train_loss, float), "Train loss should be float."
        assert isinstance(train_acc, float), "Train accuracy should be float."
        assert train_loss >= 0.0, "Train loss should be non-negative."
        assert 0.0 <= train_acc <= 1.0, "Train accuracy should be between 0 and 1."

        was_updated = any(not torch.equal(before, after.detach()) for before, after in zip(params_before, params_after))
        assert was_updated, "Expected at least one model parameter to change after training."

        print(f"Train loss: {train_loss:.4f}, Train acc: {train_acc:.4f}")
        print("TorchModel.train_one_epoch test passed.")

    def test_evaluate(self):
        print("Testing TorchModel.evaluate...")
        model, loader, criterion, optimizer, device = self._init_test_objects()

        model.train_one_epoch(loader, criterion, optimizer, device)
        eval_loss, eval_acc = model.evaluate(loader, criterion, device)

        assert isinstance(eval_loss, float), "Eval loss should be float."
        assert isinstance(eval_acc, float), "Eval accuracy should be float."
        assert eval_loss >= 0.0, "Eval loss should be non-negative."
        assert 0.0 <= eval_acc <= 1.0, "Eval accuracy should be between 0 and 1."
        assert model.model.training is False, "Model should be in eval mode after evaluate()."

        print(f"Eval loss: {eval_loss:.4f}, Eval acc: {eval_acc:.4f}")
        print("TorchModel.evaluate test passed.")


if __name__ == "__main__":
    test = TorchModelTest()
    test.test_parameters()
    test.test_train_one_epoch()
    test.test_evaluate()
    print_green("All TorchModel tests passed successfully!")
