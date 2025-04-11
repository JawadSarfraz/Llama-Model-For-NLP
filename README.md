# Research Paper Subject Classification

This project implements a multi-label classification system for research papers using the LLaMA-7B language model. The system is designed to classify research paper abstracts into relevant subjects.

## Project Structure

```
.
├── data/
│   ├── processed/           # Processed datasets
│   │   ├── train.json      # Training data
│   │   ├── val.json        # Validation data
│   │   └── test.json       # Test data
│   ├── sample_data.json    # Raw dataset
│   ├── prepare_training_data.py  # Data preparation script
│   └── get_frequent_subjects.py  # Subject analysis script
├── train_model.py          # Model training script
├── check_llama.py          # Script to check LLaMA installation
├── requirements.txt        # Project dependencies
└── README.md              # This file
```

## Dataset

The dataset consists of research papers with their abstracts and subjects. Key statistics:
- Total papers: 5,000
- Average subjects per paper: 9.19
- Unique subjects: 3,983
- Frequent subjects (20+ papers): 345

## Model Training

The project uses the following setup for training:

### Base Model
- Model: LLaMA-7B (huggyllama/llama-7b)
- Precision: 4-bit quantization with double quantization
- Framework: Hugging Face Transformers

### Fine-tuning Method
- Technique: PEFT/LoRA (Parameter-Efficient Fine-Tuning)
- LoRA Configuration:
  - Rank: 16
  - Alpha: 32
  - Target Modules: q_proj, v_proj
  - Dropout: 0.05

### Training Parameters
- Epochs: 3
- Batch Size: 2 (with gradient accumulation of 4 steps)
- Max Sequence Length: 512
- Learning Rate: Default from Trainer
- Weight Decay: 0.01
- Warmup Steps: 100

## Setup and Installation

1. Create and activate virtual environment:
```bash
python3 -m venv modelenv
source modelenv/bin/activate
```

2. Install dependencies:
```bash
pip3 install -r requirements.txt
```

3. Check LLaMA installation:
```bash
python3 check_llama.py
```

## Usage

1. Prepare the dataset:
```bash
python3 data/prepare_training_data.py
```

2. Start training:
```bash
python3 train_model.py
```

The training process will:
- Load the LLaMA-7B model with 4-bit quantization
- Apply LoRA configuration
- Train on the prepared dataset
- Save checkpoints every 100 steps
- Save the final model and LoRA adapter in the `results` directory

## Results

Training results and model checkpoints are saved in the `results` directory:
- `results/final_model`: The complete fine-tuned model
- `results/lora_adapter`: The LoRA adapter weights
- `logs`: Training logs for TensorBoard visualization

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
- Total examples: 100 (filtered to 54 for training)
- Training set: 45 examples (70%)
- Validation set: 9 examples (15%)
- Test set: 9 examples (15%)
- Data filtering: Only papers with subjects appearing 20+ times
- Source: Research paper abstracts with subject classifications

### Performance Metrics
- Initial training loss: 2.5322
- Final training loss: 2.553578249613444
- Training speed:
  - 0.385 samples per second
  - 0.043 steps per second
  - Average time per step: 23.40 seconds
- Total training time: ~352 seconds (5 minutes 52 seconds)

### Loss Interpretation
- Loss range: 0.0 (perfect) to ∞ (random)
- Current loss (2.55) indicates:
  - Better than random (which would be ~4-5)
  - Not perfect (which would be 0)
  - Reasonable for multi-label classification with limited data
- Optimal target loss for our task: 1.5-2.0
- Acceptable loss range: 2.0-2.5

### Training Process
- Steps: 15/15 completed
- Epochs: 2.87 (almost 3 full passes)
- Learning rate: 6.666666666666667e-05
- Gradient norm: 0.4271750748157501
- Memory usage: Optimized with 4-bit quantization

### Model Performance
The model shows moderate performance in:
- Identifying multiple subjects per abstract
- Handling subject relationships
- Working with limited training data
- Managing memory constraints

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
- Multi-label subject classification
- Efficient fine-tuning with LoRA
- Comprehensive logging system
- GPU memory monitoring
- Automatic checkpoint saving
- Progress tracking with tqdm
- Detailed performance metrics
- Base model testing capabilities

## Base Model Testing
Before fine-tuning, you can test the base model's classification capabilities:
```bash
python3 test_base_model.py
```
This will:
- Load a base language model (currently GPT-2 for testing)
- Test classification on a sample abstract
- Log results in the `logs` directory
- Use 4-bit quantization for memory efficiency

The base model test helps establish a performance baseline before fine-tuning.

### Future Improvements
- Replace GPT-2 with a more advanced model (Llama 7B, Mistral, etc.)
- Implement proper authentication for accessing gated models
- Optimize memory usage for larger models
- Add support for batch processing of abstracts

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

## Interactive Abstract Classifier

The project now includes an interactive command-line interface for classifying research paper abstracts:

```bash
python3 abstract_classifier.py
```

Features:
- Interactive command-line interface
- Real-time subject classification
- Supports multiple abstract inputs
- Logs all classifications for reference
- Uses the OPT-125M model for efficient processing
- Provides clear feedback and results

Usage:
1. Run the script: `python3 abstract_classifier.py`
2. Enter your abstract when prompted
3. View the predicted subjects
4. Type 'quit' to exit

Example:
```
Welcome to the Abstract Subject Classifier!
Type 'quit' to exit the program.

==================================================

Please enter your abstract (or 'quit' to exit):
[Your abstract here]

Analyzing abstract...

Results:
Abstract: [Your abstract]
Predicted Subjects: [List of subjects]
```

The classifier uses the same model architecture and classification logic as the main system, but provides a more user-friendly interface for quick testing and analysis.

# Abstract Subject Classifier

This project provides tools for classifying research abstracts into subject areas using large language models.

## Versions

### 1. OPT-125m Version (`abstract_classifier.py`)
- Uses Facebook's OPT-125m model
- Basic classification capabilities
- Suitable for quick testing and smaller abstracts

### 2. Mistral-7B Version (`abstract_classifier_mistral.py`)
- Uses Mistral-7B-Instruct-v0.2 model
- More powerful classification capabilities
- Better handling of complex abstracts
- Supports both German and English terminology
- Uses 4-bit quantization for efficient memory usage

## Features

- Classifies research abstracts into predefined subject areas
- Maintains both German and English terminology
- Logs classification results with timestamps
- Interactive command-line interface
- Caches model files locally for faster subsequent runs

## Subject Areas

The classifier recognizes the following main subject areas:
- Banking
- Trade
- EU Integration

Each area has associated terms in both German and English.

## Usage

1. Install required dependencies:
```bash
pip install torch transformers bitsandbytes
```

2. Run the classifier:
```bash
# For OPT-125m version
python3 abstract_classifier.py

# For Mistral-7B version
python3 abstract_classifier_mistral.py
```

3. Enter your abstract when prompted
4. Type 'quit' to exit the program

## Logging

Classification results are logged to the `logs` directory with timestamps for future reference.

## Requirements

- Python 3.8+
- PyTorch
- Transformers
- BitsAndBytes (for Mistral-7B version)
- Sufficient GPU memory (recommended for Mistral-7B version)