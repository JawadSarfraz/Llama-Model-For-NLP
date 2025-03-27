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
│   │   └── model.py       # Model architecture with PEFT
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

## Model Architecture

### Base Model
- Model: LLaMA-7B (Meta AI)
- Architecture: Transformer-based language model
- Base model size: 7 billion parameters
- Quantization: 4-bit with double quantization
- Compute dtype: float16
- Quantization type: nf4 (NormalFloat4)

### Fine-tuning Configuration
- Method: Parameter-Efficient Fine-Tuning (PEFT) with LoRA
- LoRA Parameters:
  - Rank (r): 16
  - Alpha scaling: 32
  - Target modules: q_proj, k_proj, v_proj, o_proj
  - Dropout: 0.05
  - Task type: SEQ_CLS (Sequence Classification)

### Memory Optimization
- 4-bit quantization for reduced memory footprint
- Double quantization for additional memory savings
- Gradient checkpointing enabled
- Mixed precision training (fp16)
- Automatic device mapping for optimal resource utilization

### Training Configuration
- Batch size: 2 (with gradient accumulation)
- Learning rate: 2e-5
- Weight decay: 0.01
- Number of epochs: 3
- Warmup steps: 100
- Evaluation strategy: Per epoch
- Save strategy: Per epoch
- Gradient clipping: 1.0
- Model checkpointing: Keep best 3 checkpoints
- Metrics: F1-score, precision, recall

### Data Processing
- Maximum sequence length: 512 tokens
- Tokenization: LLaMA tokenizer
- Padding: Max length
- Truncation: Enabled
- Label encoding: Multi-label binarization
- Dataset splits:
  - Training: 70%
  - Validation: 15%
  - Test: 15%

## Training Results

### Dataset Statistics
- Total examples: 1000 (filtered to include only papers with subjects)
- Training set: 700 examples (70%)
- Validation set: 150 examples (15%)
- Test set: 150 examples (15%)
- Data filtering: Only papers with valid subject fields are included
- Source: Research paper abstracts with subject classifications

### Performance Metrics
- Final training loss: 1.2969
- Evaluation metrics:
  - F1 Score: 0.3442 (34.42%)
  - Precision: 0.2667 (26.67%)
  - Recall: 0.5500 (55.00%)

### Training Duration
- Total training time: ~2.5 minutes
- Average time per epoch: ~45 seconds
- Checkpoint saving: Every epoch
- Model size after training: ~9.8GB (including all dependencies)

## Features

- Multi-label classification using LLaMA-7B model
- Parameter-Efficient Fine-Tuning (PEFT) with LoRA
- 4-bit quantization for memory efficiency
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
- Fine-tuning: PEFT with LoRA adapters
- Task: Multi-label classification
- Input: Paper abstracts
- Output: Subject category predictions
- Memory optimization: 
  - 4-bit quantization
  - LoRA for efficient fine-tuning
- Training metrics: F1-score, precision, recall

## Requirements

### Hardware Requirements
- GPU with at least 16GB VRAM
- 32GB system RAM recommended
- CUDA 11.7 compatible

### Software Requirements
- Python 3.10+
- PyTorch 2.0.1
- Transformers 4.49.0
- PEFT 0.14.0
- NumPy 1.21.5+
- scikit-learn 1.0.2+
- PyYAML 5.4.1+
- Pytest 7.0.0+
- Black 23.3.0+
- isort 5.12.0+
- flake8 6.0.0+
- bitsandbytes 0.41.1+
- accelerate 0.21.0+
- joblib 1.3.0+

## License

[Your License Here]

## Testing HuggingFace Setup

The project includes a test script `test_huggingface_setup.py` to verify HuggingFace installation and functionality:

```bash
python3 test_huggingface_setup.py
```

The test script verifies:
1. **Installation Check**
   - Transformers library version
   - PyTorch installation and CUDA availability
   - Required dependencies

2. **Authentication**
   - HuggingFace token validation
   - API access verification
   - User authentication status

3. **Model Operations**
   - Model loading capabilities
   - Tokenizer functionality
   - Basic inference testing
   - Performance metrics

### Test Script Output Example
```
HuggingFace Setup Test Script
Time: 2024-03-14 10:00:00

=====================================
 1. Testing HuggingFace Installation 
=====================================
✓ transformers version: 4.49.0
✓ PyTorch version: 2.0.1
✓ CUDA available: True
...
```

### Prerequisites for Testing
1. Valid HuggingFace token in `.env` file
2. Python environment with required packages
3. Internet connection for model downloading

## Subject Classification with LLaMA-7B

This project implements a multi-label subject classifier using the LLaMA-7B model with 4-bit quantization and LoRA fine-tuning.

## Project Structure

```
project/
├── configs/
│   └── config.yaml        # Configuration management
├── src/
│   ├── data/
│   │   └── data_loader.py # Data handling
│   ├── models/
│   │   └── model.py      # Model architecture
│   └── train_model.py    # Training pipeline
├── test_huggingface_setup.py  # Testing infrastructure
├── predict_subjects.py   # Script for predicting subjects
└── requirements.txt      # Dependencies
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

3. Set up Hugging Face token:
```bash
export HUGGINGFACE_TOKEN=your_token_here
```

## Training

To train the model:
```bash
python3 -m src.train_model
```

## Predicting Subjects

To predict subjects from an abstract:

```bash
python3 predict_subjects.py --abstract "Your abstract text here"
```

Example:
```bash
python3 predict_subjects.py --abstract "This paper presents a novel quantum computing algorithm for solving complex mathematical problems using advanced physics principles."
```

The script will output the predicted subjects for the given abstract.

## Model Configuration

The model uses:
- LLaMA-7B base model
- 4-bit quantization for memory efficiency
- LoRA for parameter-efficient fine-tuning
- Multi-label classification for subject prediction

## Testing

To test the Hugging Face setup:
```bash
python3 test_huggingface_setup.py
```

## Requirements

- Python 3.8+
- PyTorch
- Transformers
- PEFT
- Other dependencies listed in requirements.txt