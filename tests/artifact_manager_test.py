from unittest.mock import Mock

import pytest

from src.common.constants import Constants as consts
from src.models.base_model import BaseModel
from src.training.artifact_manager import ArtifactManager


@pytest.fixture
def patched_constants_dirs(tmp_path, monkeypatch):
    """Patch Constants.models_dir and Constants.params_dir to temporary directories."""
    models_dir = tmp_path / "models"
    params_dir = tmp_path / "params"
    monkeypatch.setattr(consts, "models_dir", models_dir)
    monkeypatch.setattr(consts, "params_dir", params_dir)
    return models_dir, params_dir


@pytest.fixture
def artifact_manager(patched_constants_dirs):
    """ArtifactManager using patched constants dirs."""
    return ArtifactManager(experiment_name="test_experiment")


@pytest.fixture
def mock_base_model():
    mock_model = Mock(spec=BaseModel)
    mock_model.save = Mock()
    mock_model.load = Mock()
    return mock_model


def test_init_sets_experiment_name():
    manager = ArtifactManager(experiment_name="my_experiment")
    assert manager.experiment_name == "my_experiment"


def test_increase_file_number_nonexistent_file(artifact_manager, tmp_path):
    test_path = tmp_path / "test_file.pkl"
    result = artifact_manager._increase_file_number(test_path)
    assert result == test_path


def test_increase_file_number_existing_file(artifact_manager, tmp_path):
    test_file = tmp_path / "test_file.pkl"
    test_file.touch()
    result = artifact_manager._increase_file_number(test_file)
    assert result.name == "test_file_0.pkl"


def test_increase_file_number_multiple_increments(artifact_manager, tmp_path):
    test_file = tmp_path / "test_file.pkl"
    test_file.touch()
    (tmp_path / "test_file_0.pkl").touch()
    (tmp_path / "test_file_1.pkl").touch()
    result = artifact_manager._increase_file_number(test_file)
    assert result.name == "test_file_2.pkl"


def test_save_params_increments_filename_on_collision(artifact_manager, patched_constants_dirs):
    """Sprawdza funkcjonalność: gdy plik istnieje, dodaje sufiks _0, potem _1, ..."""
    _, params_dir = patched_constants_dirs

    artifact_manager.save_params(params={"lr": 0.001}, file_name="my_params")
    first_path = params_dir / "test_experiment" / "my_params.pkl"
    assert first_path.exists()

    artifact_manager.save_params(params={"lr": 0.002}, file_name="my_params")
    second_path = params_dir / "test_experiment" / "my_params_0.pkl"
    assert second_path.exists()
    assert artifact_manager.changed_file_name is True

    # Po kolizji get_params_file_path powinien wskazywać ostatnio utworzony plik.
    assert artifact_manager.get_params_file_path(file_name="my_params") == second_path

    artifact_manager.save_params(params={"lr": 0.003}, file_name="my_params")
    third_path = params_dir / "test_experiment" / "my_params_1.pkl"
    assert third_path.exists()
    assert artifact_manager.get_params_file_path(file_name="my_params") == third_path


def test_save_and_load_params_roundtrip(artifact_manager, patched_constants_dirs):
    _, params_dir = patched_constants_dirs

    test_params = {"lr": 0.001, "epochs": 100}
    artifact_manager.save_params(params=test_params, file_name="roundtrip")
    saved_path = params_dir / "test_experiment" / "roundtrip.pkl"
    assert saved_path.exists()

    loaded = artifact_manager.load_params(file_name="roundtrip")
    assert loaded == test_params


def test_save_model_creates_file_and_calls_model_save(artifact_manager, patched_constants_dirs, mock_base_model):
    models_dir, _ = patched_constants_dirs

    artifact_manager.save_model(model=mock_base_model, model_name="model", ext="pt")
    expected_path = models_dir / "test_experiment" / "model.pt"
    assert expected_path.exists()
    mock_base_model.save.assert_called_once_with(file_path=expected_path)


def test_load_model_calls_model_load_with_expected_path(artifact_manager, patched_constants_dirs, mock_base_model):
    models_dir, _ = patched_constants_dirs
    exp_dir = models_dir / "test_experiment"
    exp_dir.mkdir(parents=True, exist_ok=True)
    model_path = exp_dir / "model.pt"
    model_path.touch()

    artifact_manager.load_model(model=mock_base_model, model_name="model", ext="pt")
    mock_base_model.load.assert_called_once_with(file_path=model_path)
