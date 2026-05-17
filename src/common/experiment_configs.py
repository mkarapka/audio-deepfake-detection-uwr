from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.training.objectives import Objective
    from src.models.base_model import BaseModel


@dataclass
class ExperimentPreprocessConfig:
    splits_names: list[str]
    fraction: float
    use_audio_id_sampling: bool
    balance_splits_strategy: dict[str, list] | None
    use_standardize: bool
    remove_by_query: str | dict[str, str] | None

    def get_dict(self):
        return asdict(self)


@dataclass
class TorchParameters:
    batch_size: int
    num_workers: int
    epochs: int
    use_pos_weight: bool = True


@dataclass
class OptunaTrainingConfig:
    objectives: list[Objective]
    n_trials: int
    torch_params: TorchParameters | None

@dataclass
class FinalTrainConfig:
    best_params_artifact_alias: str
    best_params_artifact_type: str
    models: list[BaseModel]
    torch_params: TorchParameters

@dataclass
class ExperimentConfig:
    preprocess_configs: dict[str, ExperimentPreprocessConfig]
    training_config: OptunaTrainingConfig | FinalTrainConfig

@dataclass
class ExperimentInfo:
    experiment_name: str
    models: list[str]
    description: str
    config: ExperimentConfig

    def get_dict(self):
        return asdict(self)
