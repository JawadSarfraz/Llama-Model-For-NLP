import json
from collections import Counter
import random
from typing import List, Dict, Set
import os

def get_frequent_subjects(data: List[dict], min_papers: int = 20) -> Set[str]:
    """
    Get subjects that appear in at least min_papers papers.
    """
    # Count subject frequencies
    subject_counts = Counter()
    for paper in data:
        if 'subject' in paper:
            subject_counts.update(paper['subject'])
    
    # Get subjects that appear in at least min_papers
    frequent_subjects = {subject for subject, count in subject_counts.items() 
                        if count >= min_papers}
    
    print(f"\nFound {len(frequent_subjects)} subjects that appear in {min_papers}+ papers")
    return frequent_subjects

def format_training_example(abstract: str, subjects: List[str]) -> dict:
    """
    Format a single training example.
    """
    return {
        'input': f"Classify this research abstract into subjects:\n\n{abstract}",
        'output': f"Subjects: {', '.join(subjects)}"
    }

def prepare_dataset(data: List[dict], frequent_subjects: Set[str]) -> List[dict]:
    """
    Prepare dataset by filtering for frequent subjects and formatting examples.
    """
    formatted_data = []
    
    for paper in data:
        if 'subject' in paper and 'abstract' in paper:
            # Get frequent subjects for this paper
            paper_subjects = [s for s in paper['subject'] if s in frequent_subjects]
            
            # Only include papers that have at least one frequent subject
            if paper_subjects:
                # Join abstract parts if it's a list
                abstract = ' '.join(paper['abstract']) if isinstance(paper['abstract'], list) else paper['abstract']
                
                # Format the example
                example = format_training_example(abstract, paper_subjects)
                formatted_data.append(example)
    
    return formatted_data

def split_dataset(data: List[dict], train_ratio=0.7, val_ratio=0.15, test_ratio=0.15) -> tuple:
    """
    Split dataset into train, validation, and test sets.
    """
    # Shuffle data
    random.shuffle(data)
    
    # Calculate split indices
    train_idx = int(len(data) * train_ratio)
    val_idx = train_idx + int(len(data) * val_ratio)
    
    # Split data
    train_data = data[:train_idx]
    val_data = data[train_idx:val_idx]
    test_data = data[val_idx:]
    
    return train_data, val_data, test_data

def main():
    # Create processed directory if it doesn't exist
    os.makedirs('data/processed', exist_ok=True)
    
    # Load the dataset
    print("Loading dataset...")
    with open('data/sample_data.json', 'r') as f:
        data = json.load(f)
    
    # Get frequent subjects
    frequent_subjects = get_frequent_subjects(data, min_papers=20)
    
    # Prepare dataset
    print("\nPreparing dataset...")
    formatted_data = prepare_dataset(data, frequent_subjects)
    print(f"Created {len(formatted_data)} training examples")
    
    # Split dataset
    print("\nSplitting dataset...")
    train_data, val_data, test_data = split_dataset(formatted_data)
    print(f"Train set: {len(train_data)} examples")
    print(f"Validation set: {len(val_data)} examples")
    print(f"Test set: {len(test_data)} examples")
    
    # Save frequent subjects
    with open('data/processed/frequent_subjects.json', 'w') as f:
        json.dump(list(frequent_subjects), f, indent=2)
    print(f"\nSaved {len(frequent_subjects)} frequent subjects to data/processed/frequent_subjects.json")
    
    # Save splits
    for split_name, split_data in [('train', train_data), 
                                  ('val', val_data), 
                                  ('test', test_data)]:
        output_file = f'data/processed/{split_name}.json'
        with open(output_file, 'w') as f:
            json.dump(split_data, f, indent=2)
        print(f"Saved {split_name} split to {output_file}")
    
    # Print example
    print("\nExample training instance:")
    print("Input:", train_data[0]['input'])
    print("Output:", train_data[0]['output'])

if __name__ == "__main__":
    # Set random seed for reproducibility
    random.seed(42)
    main() 