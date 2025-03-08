import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def extract_samples(input_file: str, output_file: str, num_samples: int = 1000):
    """Extract samples from the input JSON Lines file."""
    logger.info(f"Reading data from {input_file}")
    
    # List to store the JSON objects
    objects = []
    entries_with_subject = 0
    entries_with_abstract = 0
    entries_with_both = 0
    
    # Open and read the file line by line
    try:
        with open(input_file, 'r', encoding='utf-8') as file:
            for i, line in enumerate(file):
                if i >= num_samples:  # Stop after reading the requested number of objects
                    break
                try:
                    # Convert each line to a JSON object
                    entry = json.loads(line.strip())
                    
                    # Count statistics
                    has_subject = 'subject' in entry and entry['subject']
                    has_abstract = 'abstract' in entry and entry['abstract']
                    
                    if has_subject:
                        entries_with_subject += 1
                    if has_abstract:
                        entries_with_abstract += 1
                    if has_subject and has_abstract:
                        entries_with_both += 1
                    
                    # Only check for subject field
                    if has_subject:
                        objects.append(entry)
                        
                except json.JSONDecodeError as e:
                    logger.warning(f"Error decoding JSON on line {i+1}: {e}")
                    continue

        # Write the collected objects to a new JSON file
        with open(output_file, 'w', encoding='utf-8') as outfile:
            json.dump(objects, outfile, indent=4, ensure_ascii=False)

        logger.info(f"Successfully extracted {len(objects)} objects.")
        
        # Print detailed statistics
        logger.info(f"\nDetailed Statistics:")
        logger.info(f"Entries with subject: {entries_with_subject}")
        logger.info(f"Entries with abstract: {entries_with_abstract}")
        logger.info(f"Entries with both: {entries_with_both}")
        
        # Print some statistics about subjects
        all_subjects = set()
        for entry in objects:
            all_subjects.update(entry['subject'])
        
        logger.info(f"\nDataset Statistics:")
        logger.info(f"Total samples: {len(objects)}")
        logger.info(f"Unique subjects: {len(all_subjects)}")
        logger.info("\nSample subjects:")
        for subject in sorted(list(all_subjects))[:10]:
            logger.info(f"- {subject}")
            
    except FileNotFoundError:
        logger.error(f"File not found: {input_file}")
    except Exception as e:
        logger.error(f"An error occurred: {e}")

def main():
    # Define paths
    input_file = "/data22/stu213218/work/data/data.json"
    output_file = Path(__file__).parent.parent.parent / 'data' / 'processed' / 'expanded_dataset.json'
    
    # Extract 1000 samples
    extract_samples(input_file, str(output_file), num_samples=1000)

if __name__ == "__main__":
    main() 