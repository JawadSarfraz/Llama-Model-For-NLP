# import json

# # Specify the path to your large JSON file
# input_file_path = 'complete_data.json'
# output_file_path = 'sample_data.json'

# # List to store the first 20 JSON objects
# objects = []

# # Open and read the file line by line
# try:
#     with open(input_file_path, 'r') as file:
#         for i, line in enumerate(file):
#             if i >= 20:  # Stop after reading the first 20 objects
#                 break
#             try:
#                 # Convert each line to a JSON object and append to the list
#                 objects.append(json.loads(line.strip()))
#             except json.JSONDecodeError as e:
#                 print(f"Error decoding JSON on line {i+1}: {e}")

#     # Write the collected objects to a new JSON file
#     with open(output_file_path, 'w') as outfile:
#         json.dump(objects, outfile, indent=4)

#     print("First 20 objects extracted successfully.")
# except FileNotFoundError:
#     print(f"File not found: {input_file_path}")
# except Exception as e:
#     print(f"An error occurred: {e}")


input_file_path = 'complete_data.json'

# Test reading the first few lines of the file
try:
    with open(input_file_path, 'r') as file:
        for i in range(5):  # Adjust the range if you want to read more lines
            print(f"Line {i+1}: {file.readline().strip()}")
except FileNotFoundError:
    print(f"File not found: {input_file_path}")
except Exception as e:
    print(f"An error occurred: {e}")