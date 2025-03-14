from transformers import LlamaForSequenceClassification, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import yaml
import os

class SubjectClassifier:
    def __init__(self, config_path):
        """Initialize the model classifier with configuration"""
        print("Initializing SubjectClassifier...")
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Get the project root directory
        self.project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    def get_model(self, num_labels):
        """Initialize and return the LLaMA model for multi-label classification with PEFT"""
        print(f"\nLoading LLaMA model with {num_labels} labels...")
        
        # Configure 4-bit quantization
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=self.config['model']['quantization']['bnb_4bit_compute_dtype'],
            bnb_4bit_use_double_quant=self.config['model']['quantization']['bnb_4bit_use_double_quant'],
            bnb_4bit_quant_type=self.config['model']['quantization']['bnb_4bit_quant_type']
        )
        
        # Load base model
        model = LlamaForSequenceClassification.from_pretrained(
            self.config['model']['name'],
            quantization_config=quantization_config,
            device_map="auto",
            num_labels=num_labels,
            problem_type="multi_label_classification"
        )
        
        # Prepare model for k-bit training
        model = prepare_model_for_kbit_training(model)
        
        # Configure LoRA
        lora_config = LoraConfig(
            r=16,  # Rank
            lora_alpha=32,  # Alpha scaling
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # Target modules for LoRA
            lora_dropout=0.05,
            bias="none",
            task_type="SEQ_CLS"
        )
        
        # Get PEFT model
        model = get_peft_model(model, lora_config)
        
        # Set model configuration
        model.config.pad_token_id = model.config.eos_token_id
        
        print("Model loaded successfully with PEFT configuration!")
        return model 