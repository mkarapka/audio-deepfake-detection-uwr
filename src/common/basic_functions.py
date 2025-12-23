from datasets import load_dataset
from src.common.constants import Constants

def load_audeter_dataset(config : str, split : str | None = None):
    if Constants.data_dir.exists() is False:
        Constants.data_dir.mkdir(parents=True, exist_ok=True)
    dataset = load_dataset("wqz995/AUDETER", config, cache_dir=str(Constants.data_dir / "cache"))
    if split is None:
        return dataset
    return dataset[split]