import os
from data.data_loader import DataLoader

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
    print("Subjects:", label_encoder.inverse_transform([first_example['labels']])[0])

if __name__ == "__main__":
    test_data_loader() 