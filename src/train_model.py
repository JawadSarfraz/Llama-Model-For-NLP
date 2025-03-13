import os
import sys
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

def compute_metrics(eval_pred):
    """Compute metrics for multi-label classification."""
    predictions, labels = eval_pred
    predictions = (predictions > 0.5).astype(int)
    
    # Calculate metrics
    f1 = f1_score(labels, predictions, average='weighted')
    precision = precision_score(labels, predictions, average='weighted')
    recall = recall_score(labels, predictions, average='weighted')
    
    return {
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

def train_model():
    """Train the multi-label classification model."""
    try:
        # Get the absolute path to the config file
        current_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(os.path.dirname(current_dir), 'configs', 'config.yaml')
        
        # Initialize data loader
        logger.info("Initializing data loader...")
        data_loader = DataLoader(config_path)
        train_test_split, label_encoder = data_loader.load_and_process_data()
        
        # Initialize model
        logger.info("Initializing model...")
        model_classifier = SubjectClassifier(config_path)
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
        logger.info("Setting up trainer...")
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_test_split["train"],
            eval_dataset=train_test_split["test"],
            compute_metrics=compute_metrics,
        )
        
        # Train the model
        logger.info("Starting training...")
        trainer.train()
        
        # Save the model
        logger.info("Saving model...")
        trainer.save_model("./results/final_model")
        
        # Evaluate the model
        logger.info("Evaluating model...")
        eval_results = trainer.evaluate()
        logger.info(f"Evaluation results: {eval_results}")
        
    except Exception as e:
        logger.error(f"An error occurred during training: {e}")
        raise

if __name__ == "__main__":
    train_model()