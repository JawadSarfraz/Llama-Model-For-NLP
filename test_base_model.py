import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import logging
import os
from datetime import datetime

def setup_logging():
    """Configure logging with timestamp and level."""
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"base_model_test_{timestamp}.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

def load_model_and_tokenizer():
    """Load the base model and tokenizer."""
    logging.info("Loading base model and tokenizer...")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("facebook/opt-1.3b")
    
    # Load model with 4-bit quantization for memory efficiency
    model = AutoModelForCausalLM.from_pretrained(
        "facebook/opt-1.3b",
        load_in_4bit=True,
        device_map="auto",
        torch_dtype=torch.float16
    )
    
    logging.info("Model and tokenizer loaded successfully")
    return model, tokenizer

def classify_abstract(model, tokenizer, abstract):
    """Classify an abstract into subjects using the base model."""
    # Create a prompt that asks for subject classification
    prompt = f"""Please analyze the following research abstract and list all relevant subject areas or fields of study. 
Format your response as a comma-separated list of subjects.

Abstract: {abstract}

Subjects:"""
    
    # Tokenize the prompt
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # Generate response
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=512,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Decode and extract subjects
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Extract everything after "Subjects:"
    subjects = response.split("Subjects:")[-1].strip()
    
    return subjects

def main():
    setup_logging()
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer()
    
    # Example abstract
    abstract = """This paper presents a novel approach to machine learning-based image classification using convolutional neural networks. 
    We demonstrate improved accuracy on the CIFAR-10 dataset through architectural modifications and training techniques. 
    Our method achieves state-of-the-art results while reducing computational complexity."""
    
    # Classify the abstract
    logging.info("Classifying abstract...")
    subjects = classify_abstract(model, tokenizer, abstract)
    
    # Log results
    logging.info("Classification Results:")
    logging.info(f"Abstract: {abstract}")
    logging.info(f"Predicted Subjects: {subjects}")

if __name__ == "__main__":
    main() 