from pathlib import Path


class Constants:
    # Directories
    root = Path(__file__).parent.parent.parent
    data_dir = root / "data"
    logs_dir = data_dir / "logs"

    # File names
    extracted_embeddings_csv = "wavlm_extracted"
    embeddings_ext = "_embeddings.npy"
    metadata_ext = "_metadata.csv"

    # Audio settings
    g_sample_rate = 16_000
    wavlm_base_plus_name = "microsoft/wavlm-base-plus"

    # TTS and Vocoders configs
    tts_configs = [
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
    ]

    vocoders_configs = [
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

    tts_and_vocoders_configs = tts_configs + vocoders_configs

    # Audio types
    spoof = "spoof"
    bonafide = "bonafide"
