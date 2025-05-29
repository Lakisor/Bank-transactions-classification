# Bank Transaction Classifier

A machine learning project for classifying bank transactions using both zero-shot and fine-tuned BERT models.

## Features

- **Modular Codebase**: Clean separation of concerns with dedicated modules
- **Synthetic Data**: Realistic transaction data generation with configurable anomaly ratio
- **Dual Model Approach**:
  - Zero-shot classification using Sentence Transformers
  - Fine-tuned BERT model for improved accuracy
- **Experiment Tracking**: MLflow integration for experiment logging
- **Data Persistence**: SQLite database for storing transaction data
- **Visualization**: Confusion matrices for model comparison

## Usage

1. The script will:
   - Generate synthetic transaction data
   - Train both zero-shot and fine-tuned models
   - Save results and metrics using MLflow
   - Generate confusion matrices in the `output/` directory

## Configuration

Modify `config.py` to adjust:
- Number of samples
- Test/train split ratio
- Model parameters
- Training hyperparameters
- File paths
