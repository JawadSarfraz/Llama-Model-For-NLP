import json

# Load original JSON file
with open('econstor_2024-09-01.json', 'r') as file:
    data = json.load(file)

# Extract first 20 objects
first_20_objects = data[:20]

# Write first 20 objects to a new JSON file
with open('sample_data.json', 'w') as file:
    json.dump(first_20_objects, file, indent=4)
