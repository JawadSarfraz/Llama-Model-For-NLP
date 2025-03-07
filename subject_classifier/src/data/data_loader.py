from datasets import load_dataset
from transformers import LlamaTokenizer
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
import yaml
import os

class DataLoader:
    def __init__(self, config_path):
        """Initialize the data loader with configuration"""
        print("Initializing DataLoader...")
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        print(f"Loading tokenizer from {self.config['model']['name']}")
        self.tokenizer = LlamaTokenizer.from_pretrained(self.config['model']['name'])
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.label_encoder = MultiLabelBinarizer()
        
    def load_and_process_data(self):
        """Load and process the dataset"""
        print(f"\nLoading dataset from {self.config['data']['sample_data_path']}")
        
        # Load dataset
        dataset = load_dataset("json", data_files=self.config['data']['sample_data_path'])
        dataset = dataset.with_format("python")
        
        # Print dataset info
        print(f"Dataset size: {len(dataset['train'])} examples")
        
        # Filter out invalid examples
        dataset = dataset.filter(lambda example: "abstract" in example and example["abstract"] is not None)
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
        dataset = dataset.map(self._encode_labels)
        
        # Tokenize dataset
        print("\nTokenizing abstracts...")
        dataset = dataset.map(self._tokenize_function, batched=True)
        
        # Split dataset
        print("\nSplitting dataset into train/test...")
        train_test_split = dataset["train"].train_test_split(
            test_size=self.config['data']['test_size'],
            seed=self.config['data']['random_seed']
        )
        
        print(f"Train set size: {len(train_test_split['train'])}")
        print(f"Test set size: {len(train_test_split['test'])}")
        
        return train_test_split, self.label_encoder
    
    def _encode_labels(self, example):
        """Encode subject labels"""
        if "subject" in example and example["subject"]:
            example["labels"] = self.label_encoder.transform([example["subject"]])[0]
        else:
            example["labels"] = np.zeros(len(self.label_encoder.classes_))
        return example
    
    def _tokenize_function(self, example):
        """Tokenize text data"""
        abstract_text = " ".join([str(item) for item in example["abstract"]]) if "abstract" in example and example["abstract"] else ""
        return self.tokenizer(
            abstract_text,
            truncation=True,
            padding="max_length",
            max_length=self.config['model']['max_length']
        ) 