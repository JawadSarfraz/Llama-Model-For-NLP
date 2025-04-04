import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import joblib
import json
import logging

def setup_logging():
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def load_model_and_tokenizer():
    """Load the model and tokenizer."""
    logging.info("Loading model and tokenizer...")
    
    # Load base model and tokenizer
    model_name = "huggyllama/llama-7b"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Configure quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    
    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16
    )
    
    # Load LoRA adapter
    model = PeftModel.from_pretrained(model, "results/final_model")
    
    return model, tokenizer

def classify_abstract(abstract: str, model, tokenizer, label_encoder) -> list:
    """
    Classify an abstract into subjects.
    
    Args:
        abstract: The research abstract to classify
        model: The loaded model
        tokenizer: The loaded tokenizer
        label_encoder: The loaded label encoder
    
    Returns:
        List of predicted subjects
    """
    # Format input
    prompt = f"Classify this research abstract into subjects:\n\n{abstract}"
    
    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=512)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    # Generate prediction
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=512,
            num_return_sequences=1,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id
        )
    
    # Decode prediction
    prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract subjects from prediction
    try:
        # Find the line starting with "Subjects:"
        subjects_line = [line for line in prediction.split('\n') if line.startswith('Subjects:')][0]
        # Extract subjects and split by comma
        subjects = [s.strip() for s in subjects_line.replace('Subjects:', '').split(',')]
        return subjects
    except:
        logging.warning("Could not extract subjects from prediction")
        return []

def main():
    # Set up logging
    logger = setup_logging()
    logger.info("Starting inference...")
    
    try:
        # Load model and tokenizer
        model, tokenizer = load_model_and_tokenizer()
        
        # Load label encoder
        label_encoder = joblib.load('results/final_model/label_encoder.joblib')
        
        # Example abstract
        abstract = """
        This paper investigates the effects of home-ownership on labour mobility and unemployment duration. 
        We distinguish between finding employment locally or by being geographically mobile. We find that 
        home ownership hampers the propensity to move for job reasons but improves the chances of finding 
        local jobs, which is in accordance with the predictions from our theoretical model.
        """
        
        # Classify abstract
        logger.info("Classifying abstract...")
        subjects = classify_abstract(abstract, model, tokenizer, label_encoder)
        
        # Print results
        logger.info("Predicted subjects:")
        for subject in subjects:
            logger.info(f"- {subject}")
        
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        logger.error("Traceback:", exc_info=True)
        raise

if __name__ == "__main__":
    main() 