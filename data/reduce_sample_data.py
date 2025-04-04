import json
import random

def reduce_sample_data(input_file: str, output_file: str, num_samples: int = 500):
    """
    Reduce the sample data to a specified number of examples.
    """
    print(f"Loading data from {input_file}...")
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    print(f"Original dataset size: {len(data)} examples")
    
    # Shuffle and take first num_samples
    random.shuffle(data)
    reduced_data = data[:num_samples]
    
    print(f"Reduced dataset size: {len(reduced_data)} examples")
    
    # Save reduced dataset
    print(f"Saving reduced dataset to {output_file}...")
    with open(output_file, 'w') as f:
        json.dump(reduced_data, f, indent=2)
    
    print("Done!")

if __name__ == "__main__":
    # Set random seed for reproducibility
    random.seed(42)
    
    # Reduce sample data to 100 examples
    reduce_sample_data(
        input_file='data/sample_data.json',
        output_file='data/sample_data_100.json',
        num_samples=100
    ) 