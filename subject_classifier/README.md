# Subject Classification Model

A deep learning model for classifying academic paper abstracts into multiple subject categories using the LLaMA model.

## Project Structure

```
subject_classifier/
├── data/              # Data files
├── src/              # Source code
├── configs/          # Configuration files
├── notebooks/        # Jupyter notebooks
├── tests/           # Unit tests
└── results/         # Training results
```

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

1. Data Preparation:
```bash
python src/data/data_processor.py
```

2. Training:
```bash
python src/models/trainer.py
```

3. Inference:
```bash
python src/models/predict.py
```

## Development

- Format code: `black .`
- Sort imports: `isort .`
- Run tests: `pytest`
- Lint code: `flake8`

## License

MIT License 