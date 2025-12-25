from pathlib import Path


class Constants:
    root = Path(__file__).parent.parent.parent
    data_dir = root / "data"
    logs_dir = data_dir / "logs"
    g_sample_rate = 16_000
    wavlm_base_plus_name = "microsoft/wavlm-base-plus"
