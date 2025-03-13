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
│   ├── test_data_loader.py    # Data loading tests
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

## Dataset

The project uses a dataset of academic papers with:
- 957 academic paper abstracts
- 3,891 unique subject categories
- Multi-label classification (papers can have multiple subjects)
- Mix of English and German content
- Comprehensive metadata including titles, authors, and publication info

## Setup

1. Create a virtual environment:
```bash
python -m venv modelenv
source modelenv/bin/activate  # On Windows: modelenv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up Hugging Face token:
- Create a `.env` file in the root directory
- Add your Hugging Face token: `HF_TOKEN=your_token_here`

## Usage

### Data Processing

The project includes several data processing utilities:

1. Extract samples from source data:
```bash
python -m src.data.extract_samples
```

2. Test the data loader:
```bash
python -m src.test_data_loader
```

The data processing pipeline handles:
- JSON data loading and preprocessing
- Multi-label encoding
- Text tokenization
- Train/test splitting
- Dataset expansion and sampling
- Data validation and filtering

### Model Implementation

The model implementation features:
- LLaMA-7B based architecture
- Multi-label classification head
- 8-bit quantization for memory efficiency
- Configurable parameters via YAML

Test the model:
```bash
python -m src.test_model
```

### Training

The training pipeline includes:
- Multi-label classification training
- Evaluation metrics (F1-score, precision, recall)
- Model checkpointing
- Training monitoring and logging
- Automatic best model selection

To train the model:
```bash
python -m src.train_model
```

Training features:
- Configurable batch sizes and epochs
- Weight decay for regularization
- Automatic evaluation after each epoch
- Model checkpointing based on F1-score
- Comprehensive logging of training metrics

### Testing

The project includes a comprehensive test suite:

1. Run all tests:
```bash
pytest
```

2. Run specific test files:
```bash
pytest tests/test_data_loader.py
pytest tests/test_model.py
pytest tests/test_training.py
```

3. Run tests with verbose output:
```bash
pytest -v
```

Test coverage includes:
- Data loading and preprocessing
- Model architecture and configuration
- Training pipeline components
- Metrics computation
- Model initialization and setup

## Development

- Format code: `black .`
- Sort imports: `isort .`
- Run tests: `pytest`
- Lint code: `flake8`

## Project Status

Current implementation includes:
- ✅ Data loading and preprocessing
- ✅ Model architecture
- ✅ Basic testing infrastructure
- ✅ Dataset expansion (957 samples)
- ✅ Training pipeline with multi-label support
- ✅ Model evaluation and metrics
- ✅ Comprehensive test suite
- 🔄 Training monitoring and visualization (in progress)
- 🔄 Early stopping implementation (in progress)

## License

MIT License