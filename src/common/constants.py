from dataclasses import dataclass
from pathlib import Path


@dataclass
class AudioType:
    SPOOF: str = "spoof"
    BONAFIDE: str = "bonafide"


class Constants:
    root = Path(__file__).parent.parent.parent
    data_dir = root / "data"
    logs_dir = data_dir / "logs"
    g_sample_rate = 16_000
    wavlm_base_plus_name = "microsoft/wavlm-base-plus"

    tts_and_vocoders_configs = [
        "mls-tts-bark",
        "mls-tts-chattts",
        "mls-tts-cosyvoice",
        "mls-tts-f5_tts",
        "mls-tts-fish_speech",
        "mls-tts-sparktts",
        "mls-tts-vits",
        "mls-tts-xtts",
        "mls-tts-yourtts",
        "mls-tts-zonos",
        "mls-vocoders-bigvgan",
        "mls-vocoders-bigvsan",
        "mls-vocoders-full_band_melgan",
        "mls-vocoders-hifigan",
        "mls-vocoders-melgan",
        "mls-vocoders-multi_band_melgan",
        "mls-vocoders-parallel_wavegan",
        "mls-vocoders-style_melgan",
        "mls-vocoders-univnet",
        "mls-vocoders-vocos",
    ]

    spoof = AudioType.SPOOF
    bonafide = AudioType.BONAFIDE
