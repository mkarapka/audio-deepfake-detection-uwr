import torch
import torch.nn as nn
import torch.optim as optim
from torchmetrics.functional.classification import binary_eer

import wandb
from src.common.constants import Constants as consts
from src.common.experiment_configs import ExperimentInfo, ModelType
from src.common.logger import WandbLogger, raise_error_logger, setup_logger
from src.evaluation.binary_evaluator import BinaryEvaluator
from src.models.logistic_regression_classifier import LogisticRegressionClassifier
from src.models.mlp_classifier import MlpClassifier
from src.preprocessing.experiment_preprocessor import ExperimentPreprocessor
from src.training.artifact_manager import ArtifactManager


class FinalTrainExperiment:
    def __init__(self, *, experiment_info: ExperimentInfo, wandb_run: wandb.Run):
        self.experiment_config = experiment_info.config
        self.wandb_run = wandb_run

        self.logger = setup_logger(__class__.__name__, log_to_console=True)
        self.wandb_logger = WandbLogger(self.logger, run=self.wandb_run)

        self.training_config = self.experiment_config.training_config
        self.torch_params = self.training_config.torch_params

        self.artifact_manager = ArtifactManager(experiment_name=experiment_info.experiment_name)

    def _get_feat_suffix(self, feature_key: str) -> str:
        if "wavlm" in feature_key:
            return consts.wavlm_emb_suffix
        if "fft" in feature_key:
            return consts.fft_emb_suffix
        return "unknown"

    def _build_classifier(self, *, model_type: ModelType, best_params: dict, in_features: int):
        if model_type is ModelType.LOGISTIC_REGRESSION:
            return LogisticRegressionClassifier(input_size=in_features)

        if model_type is ModelType.MLP:
            n_layers = int(best_params.get("n_layers"))
            if n_layers is None:
                raise_error_logger(self.logger, f"Missing 'n_layers' in best_params for MLP classifier: {best_params}")
            hidden_sizes = [int(best_params[f"hidden_size_{i}"]) for i in range(n_layers)]
            dropout_rate = float(best_params["dropout_rate"])
            return MlpClassifier(
                input_size=in_features,
                hidden_sizes=hidden_sizes,
                dropout_rate=dropout_rate,
            )

        raise_error_logger(self.logger, f"Unsupported classifier for final training: {model_type.value}")

    def _train_torch_binary(
        self,
        *,
        classifier,
        train_loader,
        val_loader,
        best_params: dict,
        epochs: int,
        use_pos_weight: bool,
        log_prefix: str,
    ) -> nn.Module:
        lr = float(best_params["lr"])
        weight_decay = float(best_params["weight_decay"])

        pos_weight_value = best_params.get("pos_weight")
        pos_weight = (
            torch.tensor([float(pos_weight_value)], device=classifier.device)
            if use_pos_weight and pos_weight_value
            else None
        )

        optimizer = optim.AdamW(classifier.parameters(), lr=lr, weight_decay=weight_decay)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        best_val_eer = float("inf")
        for epoch in range(epochs):
            train_loss, train_acc = classifier.train_one_epoch(
                train_loader=train_loader,
                criterion=criterion,
                optimizer=optimizer,
                device=classifier.device,
            )

            metrics = {
                f"{log_prefix}/epoch": epoch + 1,
                f"{log_prefix}/train_loss": train_loss,
                f"{log_prefix}/train_acc": train_acc,
            }

            if val_loader is not None:
                val_loss, val_acc, y_true, y_probs = classifier.evaluate(
                    val_loader=val_loader,
                    criterion=criterion,
                    device=classifier.device,
                )
                val_eer = binary_eer(preds=y_probs, target=y_true).item()
                best_val_eer = min(best_val_eer, val_eer)
                metrics |= {
                    f"{log_prefix}/val_loss": val_loss,
                    f"{log_prefix}/val_acc": val_acc,
                    f"{log_prefix}/val_eer": val_eer,
                    f"{log_prefix}/best_val_eer": best_val_eer,
                }

            self.wandb_logger.log_metrics(metrics)

        return classifier

    def run(self):
        if self.torch_params is None:
            raise_error_logger(self.logger, "torch_params is required for final training")

        feature_type_dataloaders_map = {}
        for feature_key, preprocess_config in self.experiment_config.preprocess_configs.items():
            feat_suffix = self._get_feat_suffix(feature_key)
            preprocessor = ExperimentPreprocessor(feat_suffix=feat_suffix)

            self.wandb_logger.info(f"Preprocessing {feature_key} features with config: {preprocess_config}...")
            dataset_map = preprocessor.preprocess_data(**preprocess_config)

            self.wandb_logger.info(f"Getting Dataloaders for {feature_key} features...")
            dataloaders_map = preprocessor.get_dataloaders(dataset_map, batch_size=self.torch_params.batch_size)
            self.wandb_logger.info(f"Dataloader keys for {feature_key} features: {dataloaders_map.keys()}")

            feature_type_dataloaders_map[feature_key] = dataloaders_map

        for feature_key, dataloaders_map in feature_type_dataloaders_map.items():
            train_loader = dataloaders_map.get("train")
            dev_loader = dataloaders_map.get("dev")
            test_loader = dataloaders_map.get("test")

            if train_loader is None:
                raise_error_logger(
                    self.logger,
                    f"Missing 'train' dataloader for feature_key={feature_key}",
                )

            x_batch, _ = next(iter(train_loader))
            in_features = int(x_batch.shape[-1])

            for model_type in self.training_config.models:
                model_name = model_type.value
                params_artifact_name = f"{model_name}_{feature_key}_best_params"

                self.wandb_logger.info(
                    f"Loading best params for {model_name} / {feature_key} "
                    f"from W&B artifact {params_artifact_name}..."
                )
                best_params = self.artifact_manager.load_params_from_wandb(
                    wandb_run=self.wandb_run,
                    artifact_name=params_artifact_name,
                    artifact_type=self.training_config.best_params_artifact_type,
                    alias=self.training_config.best_params_artifact_alias,
                )

                self.wandb_logger.info(
                    f"Final training {model_name} with {feature_key} features " f"using params: {best_params}"
                )

                classifier = self._build_classifier(
                    model_type=model_type,
                    best_params=best_params,
                    in_features=in_features,
                )

                log_prefix = f"final/{model_name}/{feature_key}"

                classifier = self._train_torch_binary(
                    classifier=classifier,
                    train_loader=train_loader,
                    val_loader=dev_loader,
                    best_params=best_params,
                    epochs=self.torch_params.epochs,
                    use_pos_weight=self.torch_params.use_pos_weight,
                    log_prefix=log_prefix,
                )

                if test_loader is not None:
                    self.wandb_logger.info(f"Running final test evaluation for {model_name} / {feature_key}...")
                    evaluator = BinaryEvaluator()
                    test_metrics = evaluator.evaluate(
                        model=classifier,
                        dataloader=test_loader,
                        log_prefix=f"{log_prefix}/test",
                    )
                    self.wandb_logger.log_metrics(test_metrics)
                    self.wandb_logger.info(f"Test metrics for {model_name} / {feature_key}: {test_metrics}")

                model_artifact_name = f"{model_name}_{feature_key}_final_model"
                self.artifact_manager.save_model(model=classifier, model_name=model_artifact_name, ext="pt")
                model_path = self.artifact_manager.get_model_file_path(file_name=model_artifact_name, ext="pt")
                self.wandb_run.log_artifact(model_path, name=model_artifact_name, type="model")
                self.wandb_logger.info(f"Saved final model to {model_path} and logged to W&B.")

        self.wandb_run.finish()
