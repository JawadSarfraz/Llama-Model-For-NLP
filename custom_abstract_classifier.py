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
        
        # Configure quantization with more aggressive memory optimization
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False
        )
        
        # Load tokenizer with optimized settings
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            cache_dir=self.cache_dir,
            model_max_length=512  # Limit context length
        )
        
        # Load model with optimized settings
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map="auto",
            quantization_config=quantization_config,
            torch_dtype=torch.float16,
            cache_dir=self.cache_dir,
            low_cpu_mem_usage=True
        )
        
        self.logger.info("Model and tokenizer loaded successfully")
    
    def create_prompt(self, abstract: str) -> str:
        """Create a well-structured prompt for the model."""
        return f"""Task: Extract the main subject areas from this research abstract. List ONLY the subject areas, separated by commas.

Abstract: {abstract}

Subject areas:"""

    def classify_abstract(self, abstract: str) -> List[str]:
        """Classify an abstract into subject areas."""
        try:
            # Create and tokenize prompt
            prompt = self.create_prompt(abstract)
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=512  # Limit input length
            ).to(self.model.device)
            
            # Generate response with more controlled parameters
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=50,  # Reduced for more focused output
                    temperature=0.1,    # Lower temperature for more focused output
                    top_p=0.9,
                    do_sample=True,
                    num_return_sequences=1,
                    pad_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.5,  # Increased to prevent repetition
                    no_repeat_ngram_size=3   # Prevent repetition of phrases
                )
            
            # Process response
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract subjects after the prompt
            subjects_text = response.split("Subject areas:")[-1].strip()
            
            # Clean and process subjects
            subjects = [
                subject.strip()
                for subject in subjects_text.split(",")
                if subject.strip() and len(subject.strip()) < 50  # Filter out long responses
            ]
            
            # Limit to 5 subjects maximum
            subjects = subjects[:5]
            
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