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
    # Format input with explicit instructions
    prompt = f"""Please classify this research abstract into subjects from the following list:
{', '.join(label_encoder.classes_)}

Abstract:
{abstract}

Please respond with ONLY a line starting with "Subjects:" followed by a comma-separated list of subjects from the above list. For example:
Subjects: Employment, Labor Market, Housing Market

Your classification:"""
    
    logging.info(f"Input prompt:\n{prompt}")
    
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
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            min_length=50,  # Ensure we get enough output
            max_new_tokens=100  # Limit the output length
        )
    
    # Decode prediction
    prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)
    logging.info(f"Raw model output:\n{prediction}")
    
    # Extract subjects from prediction
    try:
        # Find the line starting with "Subjects:"
        subjects_line = [line for line in prediction.split('\n') if line.startswith('Subjects:')][0]
        # Extract subjects and split by comma
        subjects = [s.strip() for s in subjects_line.replace('Subjects:', '').split(',')]
        # Filter out empty strings and validate against available subjects
        subjects = [s for s in subjects if s and s in label_encoder.classes_]
        logging.info(f"Extracted subjects: {subjects}")
        return subjects
    except Exception as e:
        logging.warning(f"Could not extract subjects from prediction: {str(e)}")
        # Try alternative extraction
        try:
            # Look for any line containing "subjects" (case insensitive)
            subjects_lines = [line for line in prediction.split('\n') 
                            if 'subjects' in line.lower()]
            if subjects_lines:
                subjects_line = subjects_lines[0]
                # Extract everything after "subjects:" or "Subjects:"
                subjects_text = subjects_line.split(':', 1)[1].strip()
                subjects = [s.strip() for s in subjects_text.split(',')]
                # Filter out empty strings and validate against available subjects
                subjects = [s for s in subjects if s and s in label_encoder.classes_]
                logging.info(f"Extracted subjects (alternative method): {subjects}")
                return subjects
        except Exception as e2:
            logging.warning(f"Alternative extraction also failed: {str(e2)}")
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
        logger.info(f"Available subjects: {label_encoder.classes_}")
        
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
        if subjects:
            logger.info("Predicted subjects:")
            for subject in subjects:
                logger.info(f"- {subject}")
        else:
            logger.warning("No subjects were predicted")
        
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        logger.error("Traceback:", exc_info=True)
        raise

if __name__ == "__main__":
    main() 