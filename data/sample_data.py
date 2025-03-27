import json
import argparse

def extract_samples(input_file_path, output_file_path, num_samples=1000):
    """
    Extract specified number of samples from the input JSON file.
    
    Args:
        input_file_path (str): Path to the input JSON file
        output_file_path (str): Path to save the sampled data
        num_samples (int): Number of samples to extract
    """
    objects = []
    
    try:
        with open(input_file_path, 'r') as file:
            for i, line in enumerate(file):
                if i >= 1000:  # Stop after reading the first 1000 objects
                    break
                try:
                    objects.append(json.loads(line.strip()))
                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON on line {i+1}: {e}")
                    continue

        with open(output_file_path, 'w') as outfile:
            json.dump(objects, outfile, indent=4)

        print(f"Successfully extracted {len(objects)} objects.")
        print(f"Output saved to: {output_file_path}")
        print("First 1000 objects extracted successfully.")
        
    except FileNotFoundError:
        print(f"File not found: {input_file_path}")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Extract samples from large JSON file')
    parser.add_argument('--input', default='data/data.json',
                      help='Input JSON file path')
    parser.add_argument('--output', default='data/sample_data.json',
                      help='Output JSON file path')
    parser.add_argument('--samples', type=int, default=1000,
                      help='Number of samples to extract')
    
    args = parser.parse_args()
    extract_samples(args.input, args.output, args.samples)