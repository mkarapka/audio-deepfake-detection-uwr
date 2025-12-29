import functools
import time

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


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


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
