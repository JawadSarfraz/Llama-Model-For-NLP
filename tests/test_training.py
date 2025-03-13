import os
import sys
import pytest
import logging
from pathlib import Path
from transformers import Trainer, TrainingArguments
from sklearn.metrics import f1_score, precision_score, recall_score
import numpy as np

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.data_loader import DataLoader
from src.models.model import SubjectClassifier

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@pytest.fixture
def config_path():
    """Fixture to get the config file path."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(os.path.dirname(current_dir), 'configs', 'config.yaml')

@pytest.fixture
def data_loader(config_path):
    """Fixture to initialize data loader."""
    return DataLoader(config_path)

@pytest.fixture
def model_classifier(config_path):
    """Fixture to initialize model classifier."""
    return SubjectClassifier(config_path)

def test_data_loading(data_loader):
    """Test if data loading works correctly."""
    train_test_split, label_encoder = data_loader.load_and_process_data()
    
    # Check if we have train and test splits
    assert "train" in train_test_split
    assert "test" in train_test_split
    
    # Check if datasets are not empty
    assert len(train_test_split["train"]) > 0
    assert len(train_test_split["test"]) > 0
    
    # Check if label encoder is initialized
    assert label_encoder is not None
    assert len(label_encoder.classes_) > 0

def test_model_initialization(model_classifier, data_loader):
    """Test if model initialization works correctly."""
    # Get number of labels from data
    _, label_encoder = data_loader.load_and_process_data()
    num_labels = len(label_encoder.classes_)
    
    # Initialize model
    model = model_classifier.get_model(num_labels=num_labels)
    
    # Check model properties
    assert model is not None
    assert model.config.num_labels == num_labels
    assert model.config.problem_type == "multi_label_classification"

def test_training_arguments():
    """Test if training arguments are correctly configured."""
    training_args = TrainingArguments(
        output_dir="./results",
        evaluation_strategy="epoch",
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        weight_decay=0.01,
        metric_for_best_model="f1",
        load_best_model_at_end=True,
        logging_dir="./logs",
        logging_steps=10,
    )
    
    # Check training arguments
    assert training_args.output_dir == "./results"
    assert training_args.evaluation_strategy == "epoch"
    assert training_args.per_device_train_batch_size == 8
    assert training_args.num_train_epochs == 3
    assert training_args.metric_for_best_model == "f1"

def test_metrics_computation():
    """Test if metrics computation works correctly."""
    # Create sample predictions and labels
    predictions = np.array([[0.7, 0.3, 0.8], [0.2, 0.9, 0.1]])
    labels = np.array([[1, 0, 1], [0, 1, 0]])
    
    # Convert predictions to binary
    binary_predictions = (predictions > 0.5).astype(int)
    
    # Compute metrics
    f1 = f1_score(labels, binary_predictions, average='weighted')
    precision = precision_score(labels, binary_predictions, average='weighted')
    recall = recall_score(labels, binary_predictions, average='weighted')
    
    # Check if metrics are within valid range
    assert 0 <= f1 <= 1
    assert 0 <= precision <= 1
    assert 0 <= recall <= 1

def test_trainer_initialization(model_classifier, data_loader):
    """Test if trainer initialization works correctly."""
    # Get data and model
    train_test_split, label_encoder = data_loader.load_and_process_data()
    model = model_classifier.get_model(num_labels=len(label_encoder.classes_))
    
    # Set up training arguments
    training_args = TrainingArguments(
        output_dir="./results",
        evaluation_strategy="epoch",
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        weight_decay=0.01,
        metric_for_best_model="f1",
        load_best_model_at_end=True,
        logging_dir="./logs",
        logging_steps=10,
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_test_split["train"],
        eval_dataset=train_test_split["test"],
        compute_metrics=lambda eval_pred: {
            'f1': f1_score(eval_pred[1], (eval_pred[0] > 0.5).astype(int), average='weighted'),
            'precision': precision_score(eval_pred[1], (eval_pred[0] > 0.5).astype(int), average='weighted'),
            'recall': recall_score(eval_pred[1], (eval_pred[0] > 0.5).astype(int), average='weighted')
        }
    )
    
    # Check trainer properties
    assert trainer is not None
    assert trainer.args == training_args
    assert trainer.train_dataset == train_test_split["train"]
    assert trainer.eval_dataset == train_test_split["test"]

if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 