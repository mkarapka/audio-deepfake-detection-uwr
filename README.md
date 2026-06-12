# Audio Deepfake Detection — UWr

This project investigates the effectiveness of machine learning for audio deepfake detection. Developed as part of an engineering thesis at the University of Wrocław, it examines how selected feature representations—spectral features (FFT) and contextual embeddings from the pre-trained WavLM model—combined with simple classifiers (logistic regression and shallow MLPs), can detect synthetic speech in uncontrolled, real-world conditions.

**Research objectives:**

- Compare the generalization ability of **FFT** (spectral analysis) and **WavLM** (deep contextual embeddings) across different datasets
- Evaluate model robustness under domain shift using the **In-the-Wild** benchmark
- Identify optimal feature-classifier combinations that balance detection accuracy and computational efficiency
- Provide insights into which factors (TTS engine, vocoder type, temporal characteristics) influence detection performance
- Determine the minimal computational requirements for effective deepfake detection

**Full ML pipeline:**

1. Load recordings from HuggingFace datasets (or local WAV files)
2. Segment audio into 4 s chunks (2 s overlap, 16 kHz sampling rate)
3. Extract features using **FFT** (1026-dim spectral features) or **WavLM** (768-dim embeddings)
4. Apply data balancing strategies (undersample, oversample, or mixed approaches)
5. Train lightweight classifiers (**Logistic Regression**, **MLP**) with hyperparameter search (Optuna)
6. Evaluate on speaker-stratified train/dev/test splits and external **In-the-Wild** benchmark

All experiments are logged with [Weights & Biases](https://wandb.ai) for reproducibility and detailed analysis.

## Datasets

### Training (HuggingFace)

| Source | Role | HuggingFace ID |
|--------|------|----------------|
| MLS English | Bonafide (real speech) | `parler-tts/mls_eng` |
| AUDETER | Spoof (TTS + vocoders) | `wqz995/AUDETER` |
| In-the-Wild | Benchmark (real + synthetic) | `mueller91/In-The-Wild` |


## Feature extraction

| Extractor | Dimension | Description |
|-----------|-----------|-------------|
| `FFTExtractor` | 1026 | log-magnitude FFT, mean + std per frame |
| `WavLmExtractor` | 768 | `microsoft/wavlm-base-plus`, mean-pooled hidden states |

Both operate on 16 kHz mono audio. FFT features can be **standardized** (mean/std computed on the train split). WavLM embeddings are used as-is.

### Output files

After preprocessing, data is stored in `data/collected_data/`:

| File | Contents |
|------|----------|
| `feature_extracted.csv` | segment metadata |
| `feature_extracted_fft.npy` | FFT embeddings |
| `feature_extracted_wavlm.npy` | WavLM embeddings |
| `splited_data/feature_extracted_{train,dev,test}.csv` | speaker-stratified splits |

For In-the-Wild: `in_the_wild.csv`, `in_the_wild_fft.npy`, `in_the_wild_wavlm.npy`.

## Models

| Model | Framework | Optuna search space |
|-------|-----------|---------------------|
| Logistic Regression | PyTorch | lr, weight_decay, pos_weight |
| MLP | PyTorch | + number of layers, hidden sizes, dropout |

Training uses `BCEWithLogitsLoss` with optional `pos_weight`. Final training supports **early stopping** on `val_loss` (`early_stopping_patience`, `early_stopping_min_delta`).

## Metrics

Primary metric: **EER** (Equal Error Rate). Also reported: accuracy, precision, recall, F1, AUROC (at EER threshold and at 0.5).

## Project structure

```
audio-deepfake-detection-uwr/
├── src/
│   ├── common/              # constants, configs, logger, utils
│   ├── datasets/            # PyTorch AudioDataset
│   ├── evaluation/          # BinaryEvaluator
│   ├── models/              # LogisticRegression, MLP
│   ├── pipelines/
│   │   ├── preprocessing/   # feature extraction, clustering, in-the-wild
│   │   └── experiments/     # final train, final eval, hyperparam search
│   ├── preprocessing/
│   │   ├── feature_extractors/  # FFT, WavLM
│   │   ├── data_balancers/      # undersample, oversample, mix
│   │   └── io/                  # Collector, FeatureLoader
│   └── training/            # ModelTrainer, Optuna objectives, ArtifactManager
├── scripts/
│   ├── preprocessing/
│   └── experiments/
├── notebooks/
├── tests/
└── data/                    # features, models, logs (not in git)
```

## Installation

Requires **Python ≥ 3.12**.

### uv (recommended)

[uv](https://docs.astral.sh/uv/) installs dependencies and creates a virtual environment from `pyproject.toml` / `uv.lock`.

```bash
# install uv (if you don't have it yet)
curl -LsSf https://astral.sh/uv/install.sh | sh

# clone the repo and enter the directory
git clone https://github.com/mkarapka/audio-deepfake-detection-uwr.git
cd audio-deepfake-detection-uwr

# create venv and install dependencies
uv sync

# optional: dev group (pytest, black, flake8, …)
uv sync --group dev
```

Run scripts in the project environment:

```bash
uv run scripts/preprocessing/run_full_preprocessing_pipeline.py
```

Or activate the venv manually:

```bash
source .venv/bin/activate   # Linux / macOS
python scripts/preprocessing/run_full_preprocessing_pipeline.py
```

### requirements.txt (alternative)

```bash
pip install -r requirements.txt
```

### GPU

WavLM extraction and model training benefit from a GPU (CUDA or Apple MPS). Batch size is selected automatically (`get_batch_size()`).

### Weights & Biases

Before running experiments, set entity and project in `src/common/wandb_config.py` and log in:

```bash
uv run wandb login
```

## Usage

### 1. Preprocessing (training data)

```bash
# WavLM
uv run scripts/preprocessing/run_full_preprocessing_pipeline.py

# FFT
uv run scripts/preprocessing/run_full_preprocessing_pipeline_fft.py
```

Follow-up steps (after feature extraction):

```bash
uv run scripts/preprocessing/run_set_audio_ids_pipeline.py
uv run scripts/preprocessing/run_map_clusters_ids_pipeline.py
uv run scripts/preprocessing/run_split_dataset.py
```

### 2. Preprocessing (In-the-Wild)

Place files in `data/in_the_wild/audio/` and metadata in `data/in_the_wild/in_the_wild.csv`, then:

```bash
uv run scripts/preprocessing/run_in_the_wild_preprocessing.py
```

### 3. Hyperparameter search (Optuna)

```bash
uv run scripts/experiments/hyperparams_search/run_fft_vs_wavlm.py
uv run scripts/experiments/hyperparams_search/run_domain_shift.py
```

### 4. Final training + evaluation on test set (unbalanced)

```bash
uv run scripts/experiments/run_final_fft_vs_wavlm.py
uv run scripts/experiments/run_final_domain_shift.py
```

Loads best hyperparameters from W&B artifacts, trains on train/dev, evaluates on test.

### 5. Evaluation

#### A1. FFT vs. WavLM (balanced)

```bash
uv run scripts/experiments/run_final_fft_vs_wavlm_balanced.py
```

#### A2. Domain shift (balanced)
```bash
uv run scripts/experiments/run_final_domain_shift_balanced.py
```

#### B1. In-the-Wild evaluation
```bash
uv run scripts/experiments/in_the_wild/run_in_the_wild_exp.py
```

#### B2. In-the-Wild evaluation (balanced)
```bash
uv run scripts/experiments/in_the_wild/run_in_the_wild_exp_balanced.py
```

Loads trained models from W&B and evaluates on the local benchmark. For FFT, standardization uses mean/std from the `feature_extracted` train split. Class balancing (e.g. 1:1) is configured in the script.

## License

See [LICENSE](LICENSE).
