# LLaMA-7B Multi-Label Classification

This project implements a multi-label classification system using the LLaMA-7B model for scientific paper classification. The system is designed to classify papers into multiple subject categories based on their abstracts.

## Project Structure

```
.
├── configs/
│   └── config.yaml         # Configuration file
├── data/
│   └── processed/         # Processed datasets
├── src/
│   ├── data/
│   │   └── data_loader.py # Data loading and preprocessing
│   ├── model/
│   │   └── model.py       # Model architecture
│   └── train_model.py     # Training script
├── tests/
│   └── test_data_loader.py # Data loader tests
├── .env                   # Environment variables (HF_TOKEN)
├── .gitignore            # Git ignore rules
├── requirements.txt      # Python dependencies
└── README.md            # This file
```

## Setup

1. Create and activate virtual environment:
```bash
python3 -m venv modelenv
source modelenv/bin/activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Configure Hugging Face token:
- Create a `.env` file in the project root
- Add your Hugging Face token:
```
HF_TOKEN=your_token_here
```

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

The project uses a YAML configuration file (`configs/config.yaml`) for managing:
- Model parameters
- Training settings
- Data paths and splits
- Evaluation metrics

## Data

The project uses a sample dataset with the following split:
- Training: 70% of samples
- Validation: 15% of samples
- Test: 15% of samples

### Data Processing
- Abstracts are tokenized using the Llama-7B tokenizer
- Labels are encoded using MultiLabelBinarizer
- Invalid examples (missing abstracts) are filtered out

1. Training the model:
```bash
python3 -m src.train_model
```

2. Running tests:
```bash
pytest tests/
```

## Model Details

- Base model: LLaMA-7B (quantized version)
- Task: Multi-label classification
- Input: Paper abstracts
- Output: Subject category predictions

## Requirements

- Python 3.10+
- PyTorch
- Transformers
- NumPy
- PyYAML
- Pytest
- Datasets
- BitsAndBytes

## License

[Your License Here]