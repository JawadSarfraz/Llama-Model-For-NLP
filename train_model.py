import json
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
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

def main():
    # Load model and tokenizer
    print("Loading model and tokenizer...")
    model_name = "mistralai/Mistral-7B-v0.1"  # Using Mistral-7B instead of LLaMA-2
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load model in 4-bit precision
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        load_in_4bit=True,
        device_map="auto",
        torch_dtype=torch.float16
    )
    
    # Prepare model for training
    model = prepare_model_for_kbit_training(model)
    
    # Define LoRA configuration
    lora_config = LoraConfig(
        r=16,  # rank
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],  # Which modules to apply LoRA to
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )
    
    # Get PEFT model
    model = get_peft_model(model, lora_config)
    
    # Load datasets
    print("Loading datasets...")
    train_dataset = load_dataset("train")
    val_dataset = load_dataset("val")
    
    # Format datasets
    train_dataset = train_dataset.map(format_prompt)
    val_dataset = val_dataset.map(format_prompt)
    
    # Tokenize datasets
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=512
        )
    
    train_dataset = train_dataset.map(tokenize_function, batched=True)
    val_dataset = val_dataset.map(tokenize_function, batched=True)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir="results",
        num_train_epochs=3,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        warmup_steps=100,
        weight_decay=0.01,
        logging_dir="logs",
        logging_steps=10,
        evaluation_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=100,
        load_best_model_at_end=True,
        report_to="tensorboard"
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
    )
    
    # Train model
    print("Starting training...")
    trainer.train()
    
    # Save the final model
    print("Saving model...")
    trainer.save_model("results/final_model")
    
    # Save the LoRA adapter
    print("Saving LoRA adapter...")
    model.save_pretrained("results/lora_adapter")

if __name__ == "__main__":
    main() 