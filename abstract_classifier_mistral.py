import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import logging
import os
from datetime import datetime

# Set cache directory to local workspace
CACHE_DIR = os.path.join(os.getcwd(), 'model_cache')
MODEL_NAME = "bigscience/bloom-560m"  # Open access model
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
            logging.FileHandler(log_file)
        ]
    )

def load_model_and_tokenizer():
    """Load the base model and tokenizer."""
    logging.info("Loading model and tokenizer...")
    
    # Configure quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )
    
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
        cache_dir=CACHE_DIR,
        trust_remote_code=True
    )
    
    logging.info("Model and tokenizer loaded successfully")
    return model, tokenizer

def create_prompt(abstract):
    """Create a prompt for subject classification."""
    prompt = f"""Analyze the following research abstract and identify its main subject areas or fields of study.
Focus on the key topics, methodologies, and domains discussed in the abstract.

Abstract:
{abstract}

Based on the content, the main subject areas are:"""
    
    return prompt

def classify_abstract(model, tokenizer, abstract):
    """Classify an abstract into subjects using the model."""
    # Create prompt
    prompt = create_prompt(abstract)
    
    # Tokenize the prompt
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(model.device)
    
    # Generate response
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            temperature=0.7,  # Increased for more diverse outputs
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Decode the response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract the subjects from the response
    subjects = []
    if "Based on the content, the main subject areas are:" in response:
        subjects_text = response.split("Based on the content, the main subject areas are:")[1].strip()
        # Clean up the response
        subjects_text = subjects_text.split('\n')[0].strip()
        # Split by commas and clean up each subject
        subjects = [s.strip() for s in subjects_text.split(",") if s.strip()]
    
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
            # Process the abstract in one line
            abstract = ' '.join(abstract.split())
            subjects = classify_abstract(model, tokenizer, abstract)
            
            if subjects:
                print(f"\nSubjects: {', '.join(subjects)}")
                logging.info(f"Subjects: {', '.join(subjects)}")
            else:
                print("\nNo subjects could be identified.")
                logging.warning("No subjects could be identified from the abstract.")
            
        except Exception as e:
            print(f"\nError: {str(e)}")
            logging.error(f"Error processing abstract: {str(e)}")

if __name__ == "__main__":
    main() 