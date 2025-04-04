import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import logging
import os
from datetime import datetime
import huggingface_hub

# Set cache directory to local workspace
CACHE_DIR = os.path.join(os.getcwd(), 'model_cache')
MODEL_DIR = os.path.join(CACHE_DIR, 'mistral-7b')
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# Configure Hugging Face to use local cache
os.environ['TRANSFORMERS_CACHE'] = CACHE_DIR
os.environ['HF_HOME'] = CACHE_DIR

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

def download_model_files():
    """Download model files manually to local directory."""
    logging.info("Downloading model files...")
    try:
        # Download tokenizer files
        huggingface_hub.hf_hub_download(
            repo_id="mistralai/Mistral-7B-v0.1",
            filename="tokenizer.json",
            local_dir=MODEL_DIR,
            local_dir_use_symlinks=False
        )
        huggingface_hub.hf_hub_download(
            repo_id="mistralai/Mistral-7B-v0.1",
            filename="tokenizer_config.json",
            local_dir=MODEL_DIR,
            local_dir_use_symlinks=False
        )
        huggingface_hub.hf_hub_download(
            repo_id="mistralai/Mistral-7B-v0.1",
            filename="special_tokens_map.json",
            local_dir=MODEL_DIR,
            local_dir_use_symlinks=False
        )
        
        # Download model files (sharded)
        for i in range(1, 5):  # Assuming 4 shards
            huggingface_hub.hf_hub_download(
                repo_id="mistralai/Mistral-7B-v0.1",
                filename=f"model-{i:05d}-of-00004.safetensors",
                local_dir=MODEL_DIR,
                local_dir_use_symlinks=False
            )
        
        huggingface_hub.hf_hub_download(
            repo_id="mistralai/Mistral-7B-v0.1",
            filename="config.json",
            local_dir=MODEL_DIR,
            local_dir_use_symlinks=False
        )
        
        logging.info("Model files downloaded successfully")
    except Exception as e:
        logging.error(f"Error downloading model files: {e}")
        raise

def load_model_and_tokenizer():
    """Load the base model and tokenizer."""
    logging.info("Loading base model and tokenizer...")
    
    # Download model files if not already present
    if not os.path.exists(os.path.join(MODEL_DIR, "config.json")):
        download_model_files()
    
    # Load tokenizer from local directory
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    
    # Load model from local directory with 4-bit quantization
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR,
        load_in_4bit=True,
        device_map="auto",
        torch_dtype=torch.float16
    )
    
    logging.info("Model and tokenizer loaded successfully")
    return model, tokenizer

def classify_abstract(model, tokenizer, abstract):
    """Classify an abstract into subjects using the base model."""
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
            max_new_tokens=50,  # Generate up to 50 new tokens
            temperature=0.3,  # Lower temperature for more focused output
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Decode and extract subjects
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Extract everything after "Subjects (list only the fields, separated by commas):"
    subjects = response.split("Subjects (list only the fields, separated by commas):")[-1].strip()
    
    # Clean up the output
    subjects = subjects.split('\n')[0]  # Take only the first line
    subjects = subjects.split('Example')[0]  # Remove any additional examples
    subjects = subjects.strip()  # Remove any extra whitespace
    
    return subjects

def main():
    setup_logging()
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer()
    
    # Example abstract
    abstract = """This paper examines the transition process within Eastern Europe and the integration process with the EU and shows that the requirements for the transition towards a market economy overlap with the requirements for EU accession. Furthermore, the economic situation of the candidate countries is examined and it is pointed out that there is a large gap in the economic development between the Central and Eastern European countries and the EU. The paper argues that the acceding transition countries still have substantial reform tasks ahead of them and that the expanding EU membership requires also considerable reforms within the EU to reduce the danger of standstill for European policy making."""
    
    # Classify the abstract
    logging.info("Classifying abstract...")
    subjects = classify_abstract(model, tokenizer, abstract)
    
    # Log results
    logging.info("Classification Results:")
    logging.info(f"Abstract: {abstract}")
    logging.info(f"Predicted Subjects: {subjects}")

if __name__ == "__main__":
    main() 