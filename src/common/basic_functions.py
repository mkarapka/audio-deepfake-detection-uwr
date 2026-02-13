import functools
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
from datasets import load_dataset

from src.common.constants import Constants as consts
from src.common.logger import get_logger, setup_logger

logger = get_logger("basic_functions")
setup_logger("basic_functions", log_to_console=True)


def load_audeter_dataset(config: str, split: str | None = None):
    if consts.data_dir.exists() is False:
        consts.data_dir.mkdir(parents=True, exist_ok=True)
    dataset = load_dataset(consts.audeter_ds_path, config, cache_dir=str(consts.data_dir / "cache"))
    if split is None:
        return dataset
    return dataset[split]


def load_audio_dataset_by_streaming(dataset: str, config: str | None, split: str):
    if dataset is None:
        logger.error("Dataset name must be provided to load audio dataset.")
    if split is None:
        logger.error("Dataset split must be provided to load audio dataset.")
    if config is None:
        return load_dataset(dataset, split=split, streaming=True)
    return load_dataset(dataset, config, split=split, streaming=True)


def get_device(include_mps: bool = True) -> str:
    if torch.cuda.is_available():
        return "cuda"
    if include_mps and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def measure_time(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        print(f"{func.__name__} took {end - start:.4f} seconds")
        return result

    return wrapper


def generate_config_sample(seed=42, num_tts=2, num_vocoders=2):
    np.random.seed(seed)
    tts_sample = np.random.choice(consts.tts_configs, num_tts, replace=False)
    vocoders_sample = np.random.choice(consts.vocoders_configs, num_vocoders, replace=False)
    configs_lst = np.hstack([tts_sample, vocoders_sample]).tolist()
    logger.info(f"Selected configurations for preprocessing: {configs_lst}")
    return configs_lst


def get_device_name():
    device = get_device()
    if device == "cuda":
        return torch.cuda.get_device_name(0)
    elif device == "mps":
        return "Apple Silicon"
    else:
        return "CPU"


def get_batch_size():
    device = get_device()
    if device == "cuda":
        if "A100" in torch.cuda.get_device_name(0):
            return 256
        elif "L4" in torch.cuda.get_device_name(0):
            return 64
        elif "T4" in torch.cuda.get_device_name(0):
            return 32
        else:
            return 16
    elif device == "mps":
        return 8
    else:
        return 4


def plot_embeddings_2d(
    embeddings: np.ndarray,
    title: str,
    labels: list[int] | None = None,
    figsize: tuple[int, int] = (8, 8),
    alpha: float = 0.7,
    size: int = 1,
    cmap: str = "tab20",
):
    plt.figure(figsize=figsize)
    if embeddings.shape[1] != 2:
        print("Warning: Embeddings are not 2D. Plot may not be meaningful.")
    if labels is not None:
        scatter = plt.scatter(embeddings[:, 0], embeddings[:, 1], c=labels, cmap=cmap, alpha=alpha, s=size)
        plt.legend(*scatter.legend_elements(), title="Clusters")
    else:
        plt.scatter(embeddings[:, 0], embeddings[:, 1], alpha=alpha, s=size)
    plt.title(title)
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.show()


def plot_embeddings_subplots(
    embeddings: list[np.ndarray],
    titles: list[str],
    labels_list: list[list[int]] | None = None,
    subplot_size: tuple[int, int] = (5, 5),
    n_cols: int = 2,
    alpha: float = 0.7,
    size: int = 1,
    cmap: str = "tab20",
):
    if len(embeddings) != len(titles):
        raise ValueError("Number of embeddings sets must match number of titles.")
    n_rows = (len(titles) + n_cols - 1) // n_cols
    plt.figure(figsize=(n_cols * subplot_size[0], n_rows * subplot_size[1]))
    for i in range(len(titles)):
        plt.subplot(n_rows, n_cols, i + 1)
        if embeddings[i].shape[1] != 2:
            print(f"Warning: Embeddings set {i} is not 2D. Plot may not be meaningful.")
        if labels_list is not None and labels_list[i] is not None:
            scatter = plt.scatter(
                embeddings[i][:, 0], embeddings[i][:, 1], c=labels_list[i], cmap=cmap, alpha=alpha, s=size
            )
            plt.legend(*scatter.legend_elements(), title="Clusters")
        else:
            plt.scatter(embeddings[i][:, 0], embeddings[i][:, 1], alpha=alpha, s=size)
        plt.title(titles[i])
        plt.xlabel("Dimension 1")
        plt.ylabel("Dimension 2")
    plt.tight_layout()
    plt.show()


def print_green(message: str, *, end: str = "\n"):
    """Wypisuje tekst na zielono bezpośrednio do konsoli."""
    GREEN = "\033[32m"
    RESET = "\033[0m"
    print(f"{GREEN}{message}{RESET}", end=end)
