# Subject Classification Model

A deep learning model for classifying academic paper abstracts into multiple subject categories using the LLaMA model. This project implements a multi-label classification system that can predict multiple subject categories for academic papers based on their abstracts.

## Project Structure

```
.
├── data/              # Data files
│   ├── raw/          # Original data files
│   └── processed/    # Processed and tokenized data
│       ├── sample_data.json       # Initial test dataset (20 samples)
│       └── expanded_dataset.json  # Expanded dataset (957 samples)
├── src/              # Source code
│   ├── data/        # Data loading and processing
│   │   ├── data_loader.py      # Data loading and preprocessing
│   │   └── extract_samples.py  # Dataset expansion utilities
│   ├── models/      # Model implementation
│   │   └── model.py
│   └── utils/       # Utility functions
├── configs/          # Configuration files
│   └── config.yaml  # Model and training configurations
├── notebooks/        # Jupyter notebooks for analysis
├── tests/           # Unit tests
│   ├── test_data_loader.py    # Data loading and split tests
│   ├── test_model.py          # Model architecture tests
│   └── test_training.py       # Training pipeline tests
├── results/         # Training results and model checkpoints
├── requirements.txt # Project dependencies
└── README.md        # Project documentation
```

## Features

- Multi-label classification using LLaMA-7B model
- Efficient data loading and preprocessing
- Support for academic paper abstracts
- Configurable model parameters
- Comprehensive logging and monitoring
- Test suite for data loading and model verification
- Scalable dataset extraction and processing
- Multi-label metrics (F1-score, precision, recall)
- Model checkpointing and evaluation
- Comprehensive test suite for all components
- Three-way data splitting with configurable ratios
- Automated testing of data processing pipeline

## Dataset Information

The dataset contains 277,284 examples with the following characteristics:
- Total unique subjects: 250,503
- Valid examples after filtering: 234,046
- Data format: JSONL (one JSON object per line)

### Data Split
The dataset is split into three parts:
- Training set: 163,832 samples (70%)
- Validation set: 35,106 samples (15%)
- Test set: 35,108 samples (15%)

### Data Processing
- Abstracts are tokenized using the Llama-7B tokenizer
- Labels are encoded using MultiLabelBinarizer
- Invalid examples (missing abstracts) are filtered out

## Setup

1. Create a virtual environment:
```