from pathlib import Path


class Constants:
    # Directories
    root = Path(__file__).parent.parent.parent
    data_dir = root / "data"
    logs_dir = data_dir / "logs"
    collected_data_dir = data_dir / "collected_data"
    train_results_dir = data_dir / "results"
    models_dir = data_dir / "models"
    split_dir = collected_data_dir / "splited_data"
    tests_data_dir = data_dir / "tests_data"

    # File names
    speakers_ids_file = "speakers_ids.csv"
    feature_extracted = "feature_extracted"
    wavlm_emb_suffix = "_wavlm"
    fft_emb_suffix = "_fft"
    npy_ext = ".npy"
    csv_ext = ".csv"

    # Audio settings
    g_sample_rate = 16_000
    wavlm_base_plus_name = "microsoft/wavlm-base-plus"
    A_100_BATCH_SIZE = 256
    ESTIMATED_RECORDS_IN_DATASET = (
        3800  # Approximate number of records in AUDETER and MLS English datasets in single split
    )

    # Datasets paths
    audeter_ds_path = "wqz995/AUDETER"
    mls_eng_ds_path = "parler-tts/mls_eng"

    # TTS, Vocoders and MLS English configs
    mls_eng_config = "mls-bonafide"
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

    # Splits configs
    base_splits_names = ["train", "dev", "test"]
    basic_train_dev_test_config = {
        "train": 0.7,
        "dev": 0.15,
        "test": 0.15,
    }

    # Balancing configs
    ratios_config = {
        "unbalanced": [None],
        "oversample": [0.5, 0.75, 1.0],
        "undersample": [0.5, 0.75, 1.0],
        "mix": [[0.5, 0.75], [0.5, 1.0], [0.75, 1.0]],
    }
    only_equal_ratios_config = {
        "undersample": [1.0],
        "mix": [[0.5, 1.0]],
        "oversample": [1.0],
    }
    only_mix_equal_ratio_config = {"mix": [[0.5, 1.0]]}
    mix_and_unbalanced_config = {"mix": [[0.5, 1.0]], "unbalanced": [None]}

    # UMAP and HDBSCAN configs
    umap_20d_config = {
        "n_components": 20,
        "n_neighbors": 30,
        "min_dist": 0.1,
        "metric": "cosine",
        "random_state": 42,
    }
    umap_2d_config = {
        "n_components": 2,
        "n_neighbors": 30,
        "min_dist": 0.1,
        "metric": "cosine",
        "random_state": 42,
    }
    hdbscan_config = {
        "min_cluster_size": 335,
        "min_samples": 3,
        "metric": "euclidean",
        "cluster_selection_method": "eom",
        "prediction_data": True,
    }
