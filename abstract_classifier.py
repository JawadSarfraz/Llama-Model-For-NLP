import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import logging
import os
from datetime import datetime

# Set cache directory to local workspace
CACHE_DIR = os.path.join(os.getcwd(), 'model_cache')
MODEL_NAME = "facebook/opt-125m"
os.makedirs(CACHE_DIR, exist_ok=True)

# Configure Hugging Face to use local cache
os.environ['TRANSFORMERS_CACHE'] = CACHE_DIR
os.environ['HF_HOME'] = CACHE_DIR

def setup_logging():
    """Configure logging with timestamp and level."""
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"abstract_classifier_{timestamp}.log")
    
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
    logging.info("Loading model and tokenizer...")
    
    # Load tokenizer and model directly
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        load_in_4bit=True,
        device_map="auto",
        torch_dtype=torch.float16,
        cache_dir=CACHE_DIR
    )
    
    logging.info("Model and tokenizer loaded successfully")
    return model, tokenizer

def classify_abstract(model, tokenizer, abstract):
    """Classify an abstract into subjects using the model."""
    # Create a prompt that asks for subject classification with examples
    prompt = f"""Task: Analyze the following research abstract and list ONLY the main subject areas or fields of study.

Example 1:
Abstract: A deep learning approach to natural language processing with attention mechanisms.
Subjects: Computer Science, Artificial Intelligence, Natural Language Processing, Machine Learning

Example 2:
Abstract: Novel techniques in quantum computing for solving optimization problems.
Subjects: Physics, Quantum Computing, Computer Science, Mathematics

Now analyze this abstract:
{abstract}

Subjects (list only the fields, separated by commas):"""
    
    # Tokenize the prompt
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # Generate response with lower temperature for more focused output
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=50,
            temperature=0.3,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Decode and extract subjects
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    subjects = response.split("Subjects (list only the fields, separated by commas):")[-1].strip()
    subjects = subjects.split('\n')[0]
    subjects = subjects.split('Example')[0]
    subjects = subjects.strip()
    
    return subjects

def main():
    setup_logging()
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer()
    
    print("\nWelcome to the Abstract Subject Classifier!")
    print("Type 'quit' to exit the program.")
    
    while True:
        print("\n" + "="*50)
        abstract = input("\nPlease enter your abstract (or 'quit' to exit):\n")
        
        if abstract.lower() == 'quit':
            print("\nThank you for using the Abstract Subject Classifier!")
            break
        
        if not abstract.strip():
            print("Please enter a valid abstract.")
            continue
        
        try:
            print("\nAnalyzing abstract...")
            subjects = classify_abstract(model, tokenizer, abstract)
            
            print("\nResults:")
            print(f"Abstract: {abstract}")
            print(f"Predicted Subjects: {subjects}")
            
            # Log the interaction
            logging.info(f"Abstract: {abstract}")
            logging.info(f"Predicted Subjects: {subjects}")
            
        except Exception as e:
            print(f"\nError: {str(e)}")
            logging.error(f"Error processing abstract: {str(e)}")

if __name__ == "__main__":
    main() 