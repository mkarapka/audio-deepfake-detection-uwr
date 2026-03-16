# Audio Deepfake Detection - UWr

A project dedicated to detecting fake, artificially generated voice recordings (audio deepfakes / spoofing). This project is being developed as part of an engineering thesis.

**Project Status: Work in Progress (WIP)** - *The project is not yet fully completed.*

## About the Project
The goal of this solution is to build a flexible and effective machine learning pipeline capable of distinguishing real human speech from synthetic samples.

The project experiments with:
- **Processing different audio representations:** From simple transformations (FFT) to powerful, advanced models based on Transformer architectures (e.g., the **WavLM** extractor).
- **Dataset balancing:** Custom approaches and pipelines solving the problem of imbalanced classes (oversampling real data, undersampling spoofs, mix strategies).
- **ML Classifiers:** From the simplest baselines, through Logistic Regression, up to MLP (Multi-Layer Perceptron) classifiers.

## Project Structure

- `src/` – Main application source code:
  - `models/` – Implementations of classifying models (MLP networks, Logistic Regression, FFT baseline model).
  - `preprocessing/` – Modules for audio segmentation, feature extraction, and dataset splitting/balancing.
  - `pipelines/` – Complex flows for experiments and final processing (e.g., data processing pipeline, classifier training pipeline).
- `notebooks/` – Jupyter notebooks used for exploratory data analysis, testing, and visualizing clusters to help understand the audio embedding space.
- `tests/` – An extensive set of automated tests verifying the correctness of the logic and data flow.
- `data/` – Environment directory for storing raw and processed data (`.npy`, `.csv`), logs, and model structures.

## Requirements and Installation (Coming Soon)
The final list of dependencies will be available in the `requirements.txt` file. Instructions for running the project and reproducing experiments will be described in detail once all implementation tasks are finished.

