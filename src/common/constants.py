from pathlib import Path


class Constants:
    # Directories
    root = Path(__file__).parent.parent.parent
    data_dir = root / "data"
    logs_dir = data_dir / "logs"
    collected_data_dir = data_dir / "collected_data"
    training_dir = "spoof_and_bonafide"
    splited_data_dir = collected_data_dir / "splited_data"

    # File names
    speakers_ids_file = "speakers_ids.csv"
    embeddings_extension = ".npy"
    metadata_extension = ".csv"

    # Wavlm
    wavlm_file_name_prefix = "wavlm_extracted"
    embeddings_file = wavlm_file_name_prefix + embeddings_extension
    wavlm_metadata_file = wavlm_file_name_prefix + metadata_extension

    train_wavlm_file = wavlm_file_name_prefix + "_train" + metadata_extension
    dev_wavlm_file = wavlm_file_name_prefix + "_dev" + metadata_extension
    test_wavlm_file = wavlm_file_name_prefix + "_test" + metadata_extension

    # FFT
    fft_file_name_prefix = "fft_extracted"
    fft_embeddings_file = fft_file_name_prefix + embeddings_extension
    fft_metadata_file = fft_file_name_prefix + metadata_extension

    train_fft_file = fft_file_name_prefix + "_train" + metadata_extension
    dev_fft_file = fft_file_name_prefix + "_dev" + metadata_extension
    test_fft_file = fft_file_name_prefix + "_test" + metadata_extension

    # Audio settings
    g_sample_rate = 16_000
    wavlm_base_plus_name = "microsoft/wavlm-base-plus"
    A_100_BATCH_SIZE = 256
    ESTIMATED_RECORDS_IN_DATASET = (
        3800  # Approximate number of records in AUDETER and MLS English datasets in single split
    )
    mls_eng_config = "mls-bonafide"

    # Dataset paths
    audeter_ds_path = "wqz995/AUDETER"
    mls_eng_ds_path = "parler-tts/mls_eng"

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
    tts_and_vocoders_configs = tts_configs + vocoders_configs

    # Audio types
    spoof = "spoof"
    bonafide = "bonafide"

    # Configs for functions
    base_splits_names = ["train", "dev", "test"]

    basic_train_dev_test_config = {
        "train": 0.7,
        "dev": 0.15,
        "test": 0.15,
    }

    bigger_dataset_train_dev_test_config = {
        "train": 0.8,
        "dev": 0.1,
        "test": 0.1,
    }
