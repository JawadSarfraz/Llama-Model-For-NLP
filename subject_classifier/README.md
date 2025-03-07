# Subject Classification Model

A deep learning model for classifying academic paper abstracts into multiple subject categories using the LLaMA model. This project implements a multi-label classification system that can predict multiple subject categories for academic papers based on their abstracts.

## Project Structure

```
subject_classifier/
├── data/              # Data files
│   ├── raw/          # Original data files
│   └── processed/    # Processed and tokenized data
├── src/              # Source code
│   ├── data/        # Data loading and processing
│   │   └── data_loader.py
│   ├── models/      # Model implementation
│   │   └── model.py
│   └── utils/       # Utility functions
├── configs/          # Configuration files
│   └── config.yaml
├── notebooks/        # Jupyter notebooks
├── tests/           # Unit tests
└── results/         # Training results
```

## Features

- Multi-label classification using LLaMA-7B model
- Efficient data loading and preprocessing
- Support for academic paper abstracts
- Configurable model parameters
- Comprehensive logging and monitoring
- Test suite for data loading and model verification

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up Hugging Face token:
- Create a `.env` file in the root directory
- Add your Hugging Face token: `HF_TOKEN=your_token_here`

## Usage

### Data Loading

The project includes a robust data loader that handles:
- JSON data loading and preprocessing
- Multi-label encoding
- Text tokenization
- Train/test splitting

Test the data loader:
```bash
python -m src.test_data_loader
```

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

### Training (Coming Soon)

Training functionality will be implemented in the next phase, including:
- Custom training loop
- Multi-label metrics
- Model checkpointing
- Training monitoring

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
- 🔄 Training implementation (in progress)

## License

MIT License 