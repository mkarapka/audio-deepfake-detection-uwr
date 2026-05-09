from unittest.mock import Mock, patch

import joblib
import pytest

from src.models.artifact_manager import ArtifactManager
from src.models.base_model import BaseModel


@pytest.fixture
def tmp_models_dir(tmp_path):
    """Fixture for mocking models directory."""
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    return models_dir


@pytest.fixture
def tmp_params_dir(tmp_path):
    """Fixture for mocking params directory."""
    params_dir = tmp_path / "params"
    params_dir.mkdir()
    return params_dir


@pytest.fixture
def artifact_manager(tmp_path):
    """Fixture for creating ArtifactManager with mocked paths."""
    manager = ArtifactManager(experiment_name="test_experiment")
    return manager


@pytest.fixture
def mock_base_model():
    """Fixture for mocking BaseModel."""
    mock_model = Mock(spec=BaseModel)
    mock_model.save = Mock()
    mock_model.load = Mock()
    return mock_model


class TestArtifactManagerInit:
    """Tests for __init__ method."""

    def test_init_sets_experiment_name(self):
        """Test if __init__ properly sets experiment_name."""
        manager = ArtifactManager(experiment_name="my_experiment")
        assert manager.experiment_name == "my_experiment"

    def test_init_creates_logger(self):
        """Test if __init__ creates logger."""
        manager = ArtifactManager(experiment_name="my_experiment")
        assert manager.logger is not None


class TestIncreaseFileNumber:
    """Tests for _increase_file_number method."""

    def test_increase_file_number_nonexistent_file(self, artifact_manager, tmp_path):
        """Test if it returns original path if file doesn't exist."""
        test_path = tmp_path / "test_file.pkl"
        result = artifact_manager._increase_file_number(test_path)
        assert result == test_path

    def test_increase_file_number_existing_file(self, artifact_manager, tmp_path):
        """Test if it increments number if file exists."""
        test_file = tmp_path / "test_file.pkl"
        test_file.touch()

        result = artifact_manager._increase_file_number(test_file)
        assert result.name == "test_file_0.pkl"

    def test_increase_file_number_multiple_increments(self, artifact_manager, tmp_path):
        """Test if it correctly increments number for multiple files."""
        test_file = tmp_path / "test_file.pkl"
        test_file.touch()
        (tmp_path / "test_file_0.pkl").touch()
        (tmp_path / "test_file_1.pkl").touch()

        result = artifact_manager._increase_file_number(test_file)
        assert result.name == "test_file_2.pkl"

    def test_increase_file_number_without_digit_suffix(self, artifact_manager, tmp_path):
        """Test if it properly handles files without numbering."""
        test_file = tmp_path / "test_file.pkl"
        test_file.touch()

        result = artifact_manager._increase_file_number(test_file)
        assert result.name == "test_file_0.pkl"


class TestGenerateFilePath:
    """Tests for _generate_file_path method."""

    def test_generate_file_path_creates_experiment_dir(self, artifact_manager, tmp_path):
        """Test if it creates directory with experiment_name."""
        with patch("src.common.constants.Constants.models_dir", tmp_path):
            path = artifact_manager._generate_file_path(file_name="test_model", ext="pkl", main_dir=tmp_path)
            assert "test_experiment" in str(path)

    def test_generate_file_path_correct_filename(self, artifact_manager, tmp_path):
        """Test if it generates correct filename."""
        path = artifact_manager._generate_file_path(file_name="my_model", ext="pkl", main_dir=tmp_path)
        assert path.name == "my_model.pkl"

    def test_generate_file_path_creates_directory_if_not_exists(self, artifact_manager, tmp_path):
        """Test if it creates directory if it doesn't exist."""
        main_dir = tmp_path / "new_dir"
        assert not main_dir.exists()

        _ = artifact_manager._generate_file_path(file_name="test_model", ext="pkl", main_dir=main_dir)
        assert main_dir.exists()

    def test_generate_file_path_increments_existing_file(self, artifact_manager, tmp_path):
        """Test if it increments number for existing file."""
        exp_dir = tmp_path / "test_experiment"
        exp_dir.mkdir()

        existing_file = exp_dir / "model.pkl"
        existing_file.touch()

        path = artifact_manager._generate_file_path(file_name="model", ext="pkl", main_dir=tmp_path)
        assert path.name == "model_0.pkl"


class TestGetFilePath:
    """Tests for _get_file_path method."""

    def test_get_file_path_existing_file(self, artifact_manager, tmp_path):
        """Test if it retrieves path of existing file."""
        exp_dir = tmp_path / "test_experiment"
        exp_dir.mkdir()
        test_file = exp_dir / "model.pkl"
        test_file.touch()

        path = artifact_manager._get_file_path(file_name="model", ext="pkl", main_dir=tmp_path)
        assert path == test_file

    def test_get_file_path_nonexistent_file_raises_error(self, artifact_manager, tmp_path):
        """Test if it raises error for nonexistent file."""
        with pytest.raises(Exception):
            artifact_manager._get_file_path(file_name="nonexistent", ext="pkl", main_dir=tmp_path)


class TestGetModelFilePath:
    """Tests for get_model_file_path method."""

    def test_get_model_file_path_uses_get_file_path(self, artifact_manager, tmp_path):
        """Test if get_model_file_path delegates to _get_file_path."""
        exp_dir = tmp_path / "test_experiment"
        exp_dir.mkdir()
        model_file = exp_dir / "my_model.pt"
        model_file.touch()

        with patch.object(artifact_manager, "_get_file_path", return_value=model_file) as mock_get:
            result = artifact_manager.get_model_file_path(file_name="my_model", ext="pt")
            mock_get.assert_called_once()
            assert result == model_file


class TestGetParamsFilePath:
    """Tests for get_params_file_path method."""

    def test_get_params_file_path_uses_get_file_path(self, artifact_manager, tmp_path):
        """Test if get_params_file_path delegates to _get_file_path."""
        exp_dir = tmp_path / "test_experiment"
        exp_dir.mkdir()
        params_file = exp_dir / "params.pkl"
        params_file.touch()

        with patch.object(artifact_manager, "_get_file_path", return_value=params_file) as mock_get:
            _ = artifact_manager.get_params_file_path(file_name="params", ext="pkl")
            mock_get.assert_called_once()

    def test_save_model_passes_correct_file_path(self, artifact_manager, mock_base_model, tmp_path):
        """Test if save_model passes correct file path to model."""
        expected_path = tmp_path / "model.pkl"

        with patch.object(artifact_manager, "get_model_file_path", return_value=expected_path):
            artifact_manager.save_model(model=mock_base_model, file_name="test_model", ext="pkl")
            mock_base_model.save.assert_called_once_with(file_path=expected_path)


class TestLoadModel:
    """Tests for load_model method."""

    def test_load_model_calls_model_load(self, artifact_manager, mock_base_model, tmp_path):
        """Test if load_model calls model.load()."""
        exp_dir = tmp_path / "test_experiment"
        exp_dir.mkdir()
        model_file = exp_dir / "model.pkl"
        model_file.touch()

        with patch.object(artifact_manager, "get_model_file_path", return_value=model_file):
            artifact_manager.load_model(model=mock_base_model, model_name="test_model", ext="pkl")
            mock_base_model.load.assert_called_once()

    def test_load_model_passes_correct_file_path(self, artifact_manager, mock_base_model, tmp_path):
        """Test if load_model passes correct file path to model."""
        expected_path = tmp_path / "model.pkl"

        with patch.object(artifact_manager, "get_model_file_path", return_value=expected_path):
            artifact_manager.load_model(model=mock_base_model, model_name="test_model", ext="pkl")
            mock_base_model.load.assert_called_once_with(file_path=expected_path)


class TestSaveParams:
    """Tests for save_params method."""

    def test_save_params_creates_file(self, artifact_manager, tmp_path):
        """Test if save_params creates params file."""
        exp_dir = tmp_path / "test_experiment"
        exp_dir.mkdir()
        params_file = exp_dir / "params.pkl"

        test_params = {"learning_rate": 0.01, "batch_size": 32}

        with patch.object(artifact_manager, "get_params_file_path", return_value=params_file):
            artifact_manager.save_params(params=test_params, file_name="test_params")
            assert params_file.exists()

    def test_save_params_saves_correct_data(self, artifact_manager, tmp_path):
        """Test if save_params saves correct data."""
        exp_dir = tmp_path / "test_experiment"
        exp_dir.mkdir()
        params_file = exp_dir / "params.pkl"

        test_params = {"learning_rate": 0.01, "batch_size": 32}

        with patch.object(artifact_manager, "get_params_file_path", return_value=params_file):
            artifact_manager.save_params(params=test_params, file_name="test_params")

            loaded_params = joblib.load(params_file)
            assert loaded_params == test_params


class TestLoadParams:
    """Tests for load_params method."""

    def test_load_params_returns_correct_data(self, artifact_manager, tmp_path):
        """Test if load_params returns correct parameters."""
        exp_dir = tmp_path / "test_experiment"
        exp_dir.mkdir()
        params_file = exp_dir / "params.pkl"

        test_params = {"learning_rate": 0.01, "batch_size": 32}
        joblib.dump(test_params, params_file)

        with patch.object(artifact_manager, "get_params_file_path", return_value=params_file):
            loaded_params = artifact_manager.load_params(file_name="test_params")
            assert loaded_params == test_params

    def test_load_params_with_complex_dict(self, artifact_manager, tmp_path):
        """Test if load_params handles complex dictionaries."""
        exp_dir = tmp_path / "test_experiment"
        exp_dir.mkdir()
        params_file = exp_dir / "params.pkl"

        test_params = {"optimizer": "adam", "lr": 0.001, "nested": {"dropout": 0.5, "layers": [128, 64, 32]}}
        joblib.dump(test_params, params_file)

        with patch.object(artifact_manager, "get_params_file_path", return_value=params_file):
            loaded_params = artifact_manager.load_params(file_name="test_params")
            assert loaded_params == test_params
            assert loaded_params["nested"]["layers"] == [128, 64, 32]


class TestArtifactManagerIntegration:
    """Integration tests for ArtifactManager."""

    def test_save_and_load_params_roundtrip(self, artifact_manager, tmp_path):
        """Test if parameters can be saved and loaded."""
        exp_dir = tmp_path / "test_experiment"
        exp_dir.mkdir()
        params_file = exp_dir / "params.pkl"

        test_params = {"lr": 0.001, "epochs": 100}

        with patch.object(artifact_manager, "get_params_file_path", return_value=params_file):
            artifact_manager.save_params(params=test_params, file_name="my_params")
            loaded_params = artifact_manager.load_params(file_name="my_params")
            assert loaded_params == test_params
