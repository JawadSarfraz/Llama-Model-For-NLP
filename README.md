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

## Configuration

The project uses a YAML configuration file (`configs/config.yaml`) for managing:

### Model Configuration
- Base model: LLaMA-7B (4-bit quantized)
- Maximum sequence length: 512 tokens
- Problem type: Multi-label classification
- Quantization settings for memory efficiency

### Training Configuration
- Output directory: `./results`
- Batch sizes: 2 (train/eval)
- Learning rate: 2e-5
- Number of epochs: 3
- Evaluation strategy: Per epoch
- Save strategy: Per epoch
- Model checkpointing: Keep best 3 checkpoints
- Metrics: F1-score, precision, recall

### Data Configuration
- Dataset splits:
  - Training: 70%
  - Validation: 15%
  - Test: 15%
- Random seed: 42
- Tokenization settings:
  - Truncation: Enabled
  - Padding: Max length

## Data Processing

The data processing pipeline includes:
1. Loading and validating JSON/JSONL data
2. Filtering invalid examples
3. Creating subject vocabulary
4. Encoding labels using MultiLabelBinarizer
5. Tokenizing abstracts with LLaMA tokenizer
6. Splitting data into train/validation/test sets

## Usage

1. Training the model:
```bash
python3 -m src.train_model
```

2. Running tests:
```bash
pytest tests/
```

## Model Details

- Base model: LLaMA-7B (4-bit quantized version)
- Task: Multi-label classification
- Input: Paper abstracts
- Output: Subject category predictions
- Memory optimization: 4-bit quantization
- Training metrics: F1-score, precision, recall

## Requirements

- Python 3.10+
- PyTorch
- Transformers
- NumPy
- PyYAML
- Pytest
- Datasets
- BitsAndBytes
- scikit-learn

## License

[Your License Here]