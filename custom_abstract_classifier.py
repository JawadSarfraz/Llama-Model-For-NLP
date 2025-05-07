import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import logging
import os
from datetime import datetime
from typing import List, Dict, Optional
import json

class AbstractClassifier:
    """A class to classify research paper abstracts into subject areas."""
    
    def __init__(
        self,
        model_name: str = "facebook/opt-350m",
        cache_dir: Optional[str] = None,
        device: str = "auto"
    ):
        self.model_name = model_name
        self.cache_dir = cache_dir or os.path.join(os.getcwd(), 'model_cache')
        self.device = device
        self.setup_logging()
        self.setup_model()
    
    def setup_logging(self) -> None:
        """Configure logging with timestamp."""
        log_dir = "logs"
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(log_dir, f"custom_classifier_{timestamp}.log")
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()  # Also log to console
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def setup_model(self) -> None:
        """Initialize the model and tokenizer with optimized settings."""
        self.logger.info(f"Loading model: {self.model_name}")
        
        # Configure quantization
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            cache_dir=self.cache_dir
        )
        
        # Load model with optimized settings
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map=self.device,
            quantization_config=quantization_config,
            torch_dtype=torch.float16,
            cache_dir=self.cache_dir
        )
        
        self.logger.info("Model and tokenizer loaded successfully")
    
    def create_prompt(self, abstract: str) -> str:
        """Create a well-structured prompt for the model."""
        return f"""[INST] You are an expert research paper classifier. Your task is to analyze the following research abstract and identify its main subject areas. Provide only the subject areas, separated by commas.

Abstract:
{abstract}

Provide 3-5 relevant subject areas that best describe this research. Be specific but concise.
[/INST]"""

    def classify_abstract(self, abstract: str) -> List[str]:
        """Classify an abstract into subject areas."""
        try:
            # Create and tokenize prompt
            prompt = self.create_prompt(abstract)
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=2048
            ).to(self.model.device)
            
            # Generate response with controlled parameters
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=100,
                    temperature=0.3,
                    top_p=0.9,
                    do_sample=True,
                    num_return_sequences=1,
                    pad_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.2
                )
            
            # Process response
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            response = response.split("[/INST]")[-1].strip()
            
            # Extract and clean subjects
            subjects = [
                subject.strip()
                for subject in response.split(",")
                if subject.strip()
            ]
            
            self.logger.info(f"Classified subjects: {subjects}")
            return subjects
            
        except Exception as e:
            self.logger.error(f"Error in classification: {str(e)}")
            return []
    
    def batch_classify(self, abstracts: List[str]) -> List[List[str]]:
        """Classify multiple abstracts in batch."""
        return [self.classify_abstract(abstract) for abstract in abstracts]

def main():
    # Initialize classifier
    classifier = AbstractClassifier()
    
    print("\nWelcome to the Advanced Abstract Classifier!")
    print("Type 'quit' to exit, 'batch' for batch processing, or paste your abstract.")
    
    while True:
        print("\n" + "="*50)
        user_input = input("\nEnter your choice (abstract/batch/quit):\n").lower()
        
        if user_input == 'quit':
            print("\nThank you for using the Advanced Abstract Classifier!")
            break
            
        elif user_input == 'batch':
            print("\nEnter abstracts one per line. Type 'done' when finished:")
            abstracts = []
            while True:
                abstract = input()
                if abstract.lower() == 'done':
                    break
                if abstract.strip():
                    abstracts.append(abstract)
            
            if abstracts:
                results = classifier.batch_classify(abstracts)
                for i, (abstract, subjects) in enumerate(zip(abstracts, results), 1):
                    print(f"\nAbstract {i}:")
                    print(f"Subjects: {', '.join(subjects)}")
            
        else:
            if not user_input.strip():
                print("Please enter a valid abstract.")
                continue
            
            subjects = classifier.classify_abstract(user_input)
            if subjects:
                print(f"\nIdentified Subjects: {', '.join(subjects)}")
            else:
                print("\nNo subjects could be identified.")

if __name__ == "__main__":
    main() 