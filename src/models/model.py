from transformers import LlamaForSequenceClassification
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
        """Initialize and return the LLaMA model for multi-label classification"""
        print(f"\nLoading LLaMA model with {num_labels} labels...")
        
        model = LlamaForSequenceClassification.from_pretrained(
            self.config['model']['name'],
            load_in_8bit=True,
            device_map="auto",
            num_labels=num_labels,
            problem_type="multi_label_classification"
        )
        
        # Set model configuration
        model.config.pad_token_id = model.config.eos_token_id
        
        print("Model loaded successfully!")
        return model 