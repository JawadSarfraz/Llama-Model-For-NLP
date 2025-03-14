import os
import sys
import logging
from pathlib import Path
from transformers import Trainer, TrainingArguments
from sklearn.metrics import f1_score, precision_score, recall_score
import numpy as np
from peft import PeftModel

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
            output_dir=data_loader.config['training']['output_dir'],
            evaluation_strategy=data_loader.config['training']['evaluation_strategy'],
            save_strategy=data_loader.config['training']['save_strategy'],
            per_device_train_batch_size=data_loader.config['training']['per_device_train_batch_size'],
            per_device_eval_batch_size=data_loader.config['training']['per_device_eval_batch_size'],
            num_train_epochs=data_loader.config['training']['num_train_epochs'],
            learning_rate=data_loader.config['training']['learning_rate'],
            weight_decay=data_loader.config['training']['weight_decay'],
            warmup_steps=data_loader.config['training']['warmup_steps'],
            logging_steps=data_loader.config['training']['logging_steps'],
            metric_for_best_model=data_loader.config['training']['metric_for_best_model'],
            load_best_model_at_end=data_loader.config['training']['load_best_model_at_end'],
            save_total_limit=data_loader.config['training']['save_total_limit'],
            greater_is_better=data_loader.config['training']['greater_is_better']
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
        
        # Save the PEFT model
        logger.info("Saving PEFT model...")
        peft_model_path = os.path.join(data_loader.config['training']['output_dir'], "final_model")
        trainer.save_model(peft_model_path)
        
        # Save the label encoder
        import joblib
        joblib.dump(label_encoder, os.path.join(peft_model_path, "label_encoder.joblib"))
        
        # Evaluate the model
        logger.info("Evaluating model...")
        eval_results = trainer.evaluate()
        logger.info(f"Evaluation results: {eval_results}")
        
    except Exception as e:
        logger.error(f"An error occurred during training: {e}")
        raise

if __name__ == "__main__":
    train_model()