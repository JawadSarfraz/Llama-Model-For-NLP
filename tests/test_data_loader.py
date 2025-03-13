import pytest
import os
import numpy as np
from src.data.data_loader import DataLoader

@pytest.fixture
def config_path():
    return "configs/config.yaml"

@pytest.fixture
def data_loader(config_path):
    return DataLoader(config_path)

def test_data_loader_initialization(data_loader):
    """Test that the data loader initializes correctly"""
    assert data_loader.tokenizer is not None
    assert data_loader.label_encoder is not None
    assert data_loader.config is not None

def test_data_splitting(data_loader):
    """Test that the data is split correctly into train/validation/test sets"""
    splits, label_encoder = data_loader.load_and_process_data()
    
    # Check that all three splits exist
    assert "train" in splits
    assert "validation" in splits
    assert "test" in splits
    
    # Check that splits are not empty
    assert len(splits["train"]) > 0
    assert len(splits["validation"]) > 0
    assert len(splits["test"]) > 0
    
    # Check that no data is lost in splitting
    total_samples = len(splits["train"]) + len(splits["validation"]) + len(splits["test"])
    assert total_samples == len(data_loader.config["data"]["splits"]["train_size"]) + \
           len(data_loader.config["data"]["splits"]["val_size"]) + \
           len(data_loader.config["data"]["splits"]["test_size"])
    
    # Check that splits are approximately correct size
    total_size = total_samples
    assert abs(len(splits["train"]) / total_size - 0.7) < 0.1  # 70% train
    assert abs(len(splits["validation"]) / total_size - 0.15) < 0.1  # 15% validation
    assert abs(len(splits["test"]) / total_size - 0.15) < 0.1  # 15% test

def test_label_encoding(data_loader):
    """Test that labels are properly encoded"""
    splits, label_encoder = data_loader.load_and_process_data()
    
    # Check that labels are encoded for all splits
    for split_name, split_data in splits.items():
        for example in split_data:
            assert "labels" in example
            assert isinstance(example["labels"], np.ndarray)
            assert len(example["labels"]) == len(label_encoder.classes_)

def test_tokenization(data_loader):
    """Test that abstracts are properly tokenized"""
    splits, _ = data_loader.load_and_process_data()
    
    # Check that all examples are tokenized
    for split_name, split_data in splits.items():
        for example in split_data:
            assert "input_ids" in example
            assert "attention_mask" in example
            assert len(example["input_ids"]) <= data_loader.config["model"]["max_length"] 