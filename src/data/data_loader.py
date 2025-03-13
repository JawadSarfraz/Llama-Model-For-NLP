from datasets import load_dataset
from transformers import LlamaTokenizer
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
import yaml
import os
import json

class DataLoader:
    def __init__(self, config_path):
        """Initialize the data loader with configuration"""
        print("Initializing DataLoader...")
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Get the project root directory
        self.project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        
        print(f"Loading tokenizer from {self.config['model']['name']}")
        self.tokenizer = LlamaTokenizer.from_pretrained(self.config['model']['name'])
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.label_encoder = MultiLabelBinarizer()
        
    def load_and_process_data(self):
        """Load and process the dataset"""
        # Get absolute path to sample data
        sample_data_path = os.path.join(self.project_root, self.config['data']['sample_data_path'])
        print(f"\nLoading dataset from {sample_data_path}")
        
        # Load dataset directly using json
        with open(sample_data_path, 'r') as f:
            data = json.load(f)
        
        # Convert to dataset format
        dataset = {"train": data}
        
        # Print dataset info
        print(f"Dataset size: {len(dataset['train'])} examples")
        
        # Filter out invalid examples
        dataset['train'] = [example for example in dataset['train'] 
                          if "abstract" in example and example["abstract"] is not None]
        print(f"Valid examples after filtering: {len(dataset['train'])}")
        
        # Create subject vocabulary
        print("\nCollecting unique subjects...")
        all_subjects = set()
        for example in dataset["train"]:
            if "subject" in example and example["subject"]:
                all_subjects.update(example["subject"])
        
        # Print subject information
        subject_list = sorted(list(all_subjects))
        print(f"Total unique subjects: {len(subject_list)}")
        print("Sample subjects:", subject_list[:5])
        
        # Fit label encoder
        self.label_encoder.fit([subject_list])
        
        # Encode labels
        print("\nEncoding labels...")
        for example in dataset["train"]:
            if "subject" in example and example["subject"]:
                example["labels"] = self.label_encoder.transform([example["subject"]])[0]
            else:
                example["labels"] = np.zeros(len(self.label_encoder.classes_))
        
        # Tokenize dataset
        print("\nTokenizing abstracts...")
        for example in dataset["train"]:
            abstract_text = " ".join([str(item) for item in example["abstract"]]) if "abstract" in example and example["abstract"] else ""
            tokenized = self.tokenizer(
                abstract_text,
                truncation=True,
                padding="max_length",
                max_length=self.config['model']['max_length']
            )
            example.update(tokenized)
        
        # Split dataset
        print("\nSplitting dataset into train/validation/test...")
        np.random.seed(self.config['data']['random_seed'])
        indices = np.random.permutation(len(dataset["train"]))
        
        # Calculate split indices
        train_size = int(len(indices) * self.config['data']['splits']['train_size'])
        val_size = int(len(indices) * self.config['data']['splits']['val_size'])
        
        train_indices = indices[:train_size]
        val_indices = indices[train_size:train_size + val_size]
        test_indices = indices[train_size + val_size:]
        
        train_data = [dataset["train"][i] for i in train_indices]
        val_data = [dataset["train"][i] for i in val_indices]
        test_data = [dataset["train"][i] for i in test_indices]
        
        train_val_test_split = {
            "train": train_data,
            "validation": val_data,
            "test": test_data
        }
        
        print(f"Train set size: {len(train_val_test_split['train'])}")
        print(f"Validation set size: {len(train_val_test_split['validation'])}")
        print(f"Test set size: {len(train_val_test_split['test'])}")
        
        return train_val_test_split, self.label_encoder 