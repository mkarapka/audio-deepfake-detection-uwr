import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.common.basic_functions import print_green
from src.common.constants import Constants as consts
from src.models.torch_model import TorchModel


class TestableTorchModel(TorchModel):
    def _create_model(self, in_features):
        return super()._create_model(in_features)

    def load(self, model_name: str, ext: str, sub_dir: str = None):
        return None

    def save(self, model_name: str, ext: str, sub_dir: str = None):
        return None


class PersistableTorchModel(TorchModel):
    def __init__(self, in_features: int, device: str = "cpu"):
        self.device = torch.device(device)
        model = self._create_model(in_features=in_features)
        super().__init__(model=model, class_name="PersistableTorchModel", include_mps=False)

    def _create_model(self, in_features: int):
        return nn.Sequential(
            nn.Linear(in_features=in_features, out_features=2),
        ).to(self.device)

    def state_dict(self):
        return self.model.state_dict()

    def load_state_dict(self, state_dict):
        return self.model.load_state_dict(state_dict)


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
        eval_loss, eval_acc, y_true, y_pred = model.evaluate(loader, criterion, device)

        assert isinstance(eval_loss, float), "Eval loss should be float."
        assert isinstance(eval_acc, float), "Eval accuracy should be float."
        assert isinstance(y_true, torch.Tensor), "Expected y_true to be a torch.Tensor."
        assert isinstance(y_pred, torch.Tensor), "Expected y_pred to be a torch.Tensor."
        assert eval_loss >= 0.0, "Eval loss should be non-negative."
        assert 0.0 <= eval_acc <= 1.0, "Eval accuracy should be between 0 and 1."
        assert model.model.training is False, "Model should be in eval mode after evaluate()."

        print(f"Eval loss: {eval_loss:.4f}, Eval acc: {eval_acc:.4f}")
        print("TorchModel.evaluate test passed.")

    def test_save(self):
        print("Testing TorchModel.save...")
        torch.manual_seed(27)

        model = PersistableTorchModel(in_features=2)
        save_dir = consts.tests_data_dir / "models"
        save_dir.mkdir(parents=True, exist_ok=True)
        file_path = save_dir / "torch_model_save_test.pt"

        model.save(file_path)

        assert file_path.exists(), "Expected saved model file to exist."
        payload = torch.load(file_path, map_location=model.device)
        assert "state_dict" in payload, "Expected 'state_dict' key in saved payload."
        assert "in_features" in payload, "Expected 'in_features' key in saved payload."
        assert payload["in_features"] == 2, "Expected in_features to match model input size."

        print("TorchModel.save test passed.")

    def test_load(self):
        print("Testing TorchModel.load...")
        torch.manual_seed(27)

        save_dir = consts.tests_data_dir / "models"
        save_dir.mkdir(parents=True, exist_ok=True)
        file_path = save_dir / "torch_model_load_test.pt"

        source_model = PersistableTorchModel(in_features=2)
        source_model.save(file_path)

        loaded_model = PersistableTorchModel(in_features=2)
        with torch.no_grad():
            for param in loaded_model.model.parameters():
                param.zero_()

        loaded_model.load(file_path)

        for source_param, loaded_param in zip(source_model.model.parameters(), loaded_model.model.parameters()):
            assert torch.allclose(
                source_param.detach(), loaded_param.detach()
            ), "Expected loaded parameters to match saved parameters."

        print("TorchModel.load test passed.")


if __name__ == "__main__":
    test = TorchModelTest()
    test.test_parameters()
    test.test_train_one_epoch()
    test.test_evaluate()
    test.test_save()
    test.test_load()
    print_green("All TorchModel tests passed successfully!")
