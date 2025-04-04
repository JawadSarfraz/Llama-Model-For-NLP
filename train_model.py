import json
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)
from peft import (
    prepare_model_for_kbit_training,
    LoraConfig,
    get_peft_model,
    TaskType
)
from datasets import Dataset
import os
from typing import Dict, List
import logging
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
import numpy as np

def load_dataset(split: str) -> Dataset:
    """Load and format dataset split."""
    with open(f'data/processed/{split}.json', 'r') as f:
        data = json.load(f)
    
    # Convert to HuggingFace dataset format
    dataset = Dataset.from_list(data)
    return dataset

def format_prompt(example: Dict) -> Dict:
    """Format the prompt for training."""
    # Combine input and output with clear separation
    text = f"{example['input']}\n\n{example['output']}"
    return {"text": text}

# Set up logging
def setup_logging():
    # Create logs directory if it doesn't exist
    os.makedirs("logs", exist_ok=True)
    
    # Create a timestamp for unique log files
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Set up file logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f"logs/training_{timestamp}.log"),
            logging.StreamHandler()
        ]
    )
    
    # Create TensorBoard writer
    writer = SummaryWriter(f"logs/tensorboard_{timestamp}")
    
    return writer

# Custom callback for logging metrics
class MetricsCallback:
    def __init__(self, writer):
        self.writer = writer
        self.metrics_history = {
            'train_loss': [],
            'eval_loss': [],
            'learning_rate': []
        }
    
    def on_init_end(self, args, state, control, **kwargs):
        """Called when the trainer is initialized."""
        pass
    
    def on_train_begin(self, args, state, control, **kwargs):
        """Called when training begins."""
        pass
    
    def on_train_end(self, args, state, control, **kwargs):
        """Called when training ends."""
        self.writer.close()
    
    def on_epoch_begin(self, args, state, control, **kwargs):
        """Called when an epoch begins."""
        pass
    
    def on_epoch_end(self, args, state, control, **kwargs):
        """Called when an epoch ends."""
        pass
    
    def on_step_begin(self, args, state, control, **kwargs):
        """Called when a training step begins."""
        pass
    
    def on_step_end(self, args, state, control, **kwargs):
        """Called when a training step ends."""
        pass
    
    def on_substep_end(self, args, state, control, **kwargs):
        """Called when a substep ends."""
        pass
    
    def on_pre_optimizer_step(self, args, state, control, **kwargs):
        """Called before the optimizer step."""
        pass
    
    def on_optimizer_step(self, args, state, control, **kwargs):
        """Called after the optimizer step."""
        pass
    
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        """Called after evaluation."""
        if metrics:
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    self.writer.add_scalar(f"eval/{key}", value, state.global_step)
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None:
            # Log to TensorBoard
            for key, value in logs.items():
                if isinstance(value, (int, float)):
                    self.writer.add_scalar(f"metrics/{key}", value, state.global_step)
            
            # Save metrics to history
            if 'loss' in logs:
                self.metrics_history['train_loss'].append(logs['loss'])
            if 'eval_loss' in logs:
                self.metrics_history['eval_loss'].append(logs['eval_loss'])
            if 'learning_rate' in logs:
                self.metrics_history['learning_rate'].append(logs['learning_rate'])
            
            # Save metrics history to file periodically
            if state.global_step % 100 == 0:
                with open('logs/metrics_history.json', 'w') as f:
                    json.dump(self.metrics_history, f)
    
    def on_save(self, args, state, control, **kwargs):
        """Called when a checkpoint is saved."""
        pass
    
    def on_prediction_step(self, args, state, control, **kwargs):
        """Called when a prediction step is performed."""
        pass

def main():
    # Set up logging
    writer = setup_logging()
    logging.info("Starting training process")
    
    try:
        # Check GPU memory
        logging.info("Checking GPU memory...")
        import torch
        for i in range(torch.cuda.device_count()):
            logging.info(f"GPU {i}: {torch.cuda.get_device_name(i)}")
            logging.info(f"Memory allocated: {torch.cuda.memory_allocated(i) / 1024**2:.2f} MB")
            logging.info(f"Memory cached: {torch.cuda.memory_reserved(i) / 1024**2:.2f} MB")
        
        # Load model and tokenizer
        logging.info("Loading model and tokenizer...")
        model_name = "huggyllama/llama-7b"
        
        # Configure quantization
        logging.info("Configuring model quantization...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        
        # Load tokenizer
        logging.info("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token
        logging.info("Tokenizer loaded successfully")
        
        # Load model with quantization
        logging.info("Loading model with 4-bit quantization...")
        logging.info("This may take a few minutes...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.float16
        )
        logging.info("Model loaded successfully")
        
        # Prepare model for training
        logging.info("Preparing model for training...")
        model = prepare_model_for_kbit_training(model)
        logging.info("Model prepared for training")
        
        # Configure LoRA
        logging.info("Configuring LoRA...")
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.CAUSAL_LM
        )
        
        # Get PEFT model
        logging.info("Getting PEFT model...")
        model = get_peft_model(model, lora_config)
        logging.info("PEFT model created successfully")
        
        # Load datasets
        logging.info("Loading datasets...")
        train_dataset = load_dataset("train")
        val_dataset = load_dataset("val")
        logging.info(f"Train dataset size: {len(train_dataset)}")
        logging.info(f"Validation dataset size: {len(val_dataset)}")
        
        # Tokenize datasets
        logging.info("Tokenizing dataset...")
        def tokenize_function(examples):
            """Tokenize the input text."""
            return tokenizer(
                examples["input"],
                padding="max_length",
                truncation=True,
                max_length=512,
                return_tensors="pt"
            )
        
        logging.info("Tokenizing training dataset...")
        tokenized_train = train_dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=train_dataset.column_names
        )
        logging.info("Training dataset tokenized")
        
        logging.info("Tokenizing validation dataset...")
        tokenized_val = val_dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=val_dataset.column_names
        )
        logging.info("Validation dataset tokenized")
        
        # Training arguments
        logging.info("Setting up training arguments...")
        training_args = TrainingArguments(
            output_dir="results",
            num_train_epochs=3,
            per_device_train_batch_size=2,
            per_device_eval_batch_size=2,
            gradient_accumulation_steps=4,
            evaluation_strategy="steps",
            eval_steps=100,
            save_strategy="steps",
            save_steps=100,
            learning_rate=2e-4,
            weight_decay=0.01,
            fp16=True,
            logging_steps=10,
            report_to="tensorboard"
        )
        
        # Initialize trainer
        logging.info("Initializing trainer...")
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_train,
            eval_dataset=tokenized_val,
            data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
            callbacks=[MetricsCallback(writer)]
        )
        
        # Train model
        logging.info("Starting training...")
        trainer.train()
        
        # Save the final model
        logging.info("Saving model...")
        trainer.save_model("results/final_model")
        
        # Save the LoRA adapter
        logging.info("Saving LoRA adapter...")
        model.save_pretrained("results/lora_adapter")
        
        # Close TensorBoard writer
        writer.close()
        logging.info("Training completed successfully")
        
    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")
        logging.error("Traceback:", exc_info=True)
        raise

if __name__ == "__main__":
    main() 