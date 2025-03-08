import os
import sys
import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.data_loader import DataLoader

def test_data_loader():
    # Get the absolute path to the config file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(os.path.dirname(current_dir), 'configs', 'config.yaml')
    
    # Initialize data loader
    data_loader = DataLoader(config_path)
    
    # Load and process data
    train_test_split, label_encoder = data_loader.load_and_process_data()
    
    # Print some sample data
    print("\nSample data from first example:")
    first_example = train_test_split['train'][0]
    print("Abstract:", first_example['abstract'])
    
    # Convert labels to numpy array for inverse transform
    labels = np.array([first_example['labels']])
    print("Subjects:", label_encoder.inverse_transform(labels)[0])

if __name__ == "__main__":
    test_data_loader() 