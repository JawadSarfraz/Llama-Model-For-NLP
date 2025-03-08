import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.model import SubjectClassifier
from src.data.data_loader import DataLoader

def test_model():
    # Get the absolute path to the config file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(os.path.dirname(current_dir), 'configs', 'config.yaml')
    
    # Initialize data loader to get number of labels
    data_loader = DataLoader(config_path)
    train_test_split, label_encoder = data_loader.load_and_process_data()
    
    # Initialize model
    model_classifier = SubjectClassifier(config_path)
    model = model_classifier.get_model(num_labels=len(label_encoder.classes_))
    
    # Print model information
    print("\nModel Information:")
    print(f"Model name: {model.config.name_or_path}")
    print(f"Number of labels: {model.config.num_labels}")
    print(f"Problem type: {model.config.problem_type}")
    print(f"Model type: {model.config.model_type}")

if __name__ == "__main__":
    test_model() 