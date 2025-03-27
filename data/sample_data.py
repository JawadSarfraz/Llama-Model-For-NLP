import json

# Specify the path to your large JSON file
input_file_path = 'data/data.json'
output_file_path = 'data/sample_data.json'

# List to store the first 1000 JSON objects
objects = []

# Open and read the file line by line
try:
    with open(input_file_path, 'r') as file:
        for i, line in enumerate(file):
            if len(objects) >= 1000:  # Stop after getting 1000 valid objects
                break
            try:
                # Convert each line to a JSON object and append to the list
                obj = json.loads(line.strip())
                objects.append(obj)
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON on line {i+1}: {e}")

    # Write the collected objects to a new JSON file
    with open(output_file_path, 'w') as outfile:
        json.dump(objects, outfile, indent=2)

    print(f"Successfully extracted {len(objects)} objects.")
    print(f"Output saved to: {output_file_path}")
    
except FileNotFoundError:
    print(f"File not found: {input_file_path}")
except Exception as e:
    print(f"An error occurred: {e}")