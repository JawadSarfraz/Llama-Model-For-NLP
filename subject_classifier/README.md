# Subject Classification Model

A deep learning model for classifying academic paper abstracts into multiple subject categories using the LLaMA model. This project implements a multi-label classification system that can predict multiple subject categories for academic papers based on their abstracts.

## Project Structure

```
subject_classifier/
├── data/              # Data files
│   ├── raw/          # Original data files
│   └── processed/    # Processed and tokenized data
│       ├── sample_data.json       # Initial test dataset (20 samples)
│       └── expanded_dataset.json  # Expanded dataset (~1000 samples)
├── src/              # Source code
│   ├── data/        # Data loading and processing
│   │   ├── data_loader.py      # Data loading and preprocessing
│   │   └── extract_samples.py  # Dataset expansion utilities
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
- Scalable dataset extraction and processing

## Dataset

The project uses a dataset of academic papers with:
- ~1000 academic paper abstracts
- 3,891 unique subject categories
- Multi-label classification (papers can have multiple subjects)
- Mix of English and German content
- Comprehensive metadata including titles, authors, and publication info

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
- ✅ Dataset expansion (957 samples)
- 🔄 Training implementation (in progress)

## License

MIT License 