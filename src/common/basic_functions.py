import functools
import time

import torch
from datasets import load_dataset

from src.common.constants import Constants


def load_audeter_dataset(config: str, split: str | None = None):
    if Constants.data_dir.exists() is False:
        Constants.data_dir.mkdir(parents=True, exist_ok=True)
    dataset = load_dataset("wqz995/AUDETER", config, cache_dir=str(Constants.data_dir / "cache"))
    if split is None:
        return dataset
    return dataset[split]


def load_audeter_ds_using_streaming(config: str, split: str):
    return load_dataset("wqz995/AUDETER", config, split=split, streaming=True)


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
