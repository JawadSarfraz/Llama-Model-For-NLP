import json

# Load the JSON data from a file
with open('../sample_data.json', 'r') as file:
    data = json.load(file)

# Count the number of objects in the JSON file
number_of_objects = len(data)

print(f"The JSON file contains {number_of_objects} objects.")