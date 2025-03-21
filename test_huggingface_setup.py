"""
Test script to demonstrate HuggingFace setup and functionality.
This script will:
1. Test HuggingFace installation
2. Verify token authentication
3. Load a model
4. Perform a simple inference
5. Test tokenizer functionality
6. Demonstrate model downloading
"""

import os
import sys
from datetime import datetime
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModel
from huggingface_hub import login, HfApi
import torch

def print_section(title):
    """Print a formatted section title"""
    print("\n" + "="*50)
    print(f" {title} ")
    print("="*50)

def test_installation():
    """Test if all required packages are installed"""
    print_section("1. Testing HuggingFace Installation")
    
    try:
        import transformers
        print(f"✓ transformers version: {transformers.__version__}")
    except ImportError:
        print("✗ transformers not installed")
        sys.exit(1)

    try:
        import torch
        print(f"✓ PyTorch version: {torch.__version__}")
        print(f"✓ CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"✓ CUDA version: {torch.version.cuda}")
    except ImportError:
        print("✗ PyTorch not installed")
        sys.exit(1)

def test_authentication():
    """Test HuggingFace authentication"""
    print_section("2. Testing HuggingFace Authentication")
    
    # Load token from .env file
    load_dotenv()
    token = os.getenv('HF_TOKEN')
    
    if not token:
        print("✗ HF_TOKEN not found in .env file")
        sys.exit(1)
    
    try:
        # Try to log in
        login(token)
        api = HfApi()
        user_info = api.whoami()
        print(f"✓ Successfully authenticated as: {user_info['name']}")
    except Exception as e:
        print(f"✗ Authentication failed: {str(e)}")
        sys.exit(1)

def test_model_loading():
    """Test loading a small model"""
    print_section("3. Testing Model Loading")
    
    try:
        # Use a small model for quick testing
        model_name = "prajjwal1/bert-tiny"
        
        print(f"Loading model: {model_name}")
        start_time = datetime.now()
        
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        
        load_time = datetime.now() - start_time
        print(f"✓ Model loaded successfully in {load_time.total_seconds():.2f} seconds")
        print(f"✓ Model size: {model.num_parameters():,} parameters")
        
        return model, tokenizer
    except Exception as e:
        print(f"✗ Model loading failed: {str(e)}")
        sys.exit(1)

def test_inference(model, tokenizer):
    """Test basic inference"""
    print_section("4. Testing Basic Inference")
    
    try:
        # Prepare input text
        text = "Testing HuggingFace functionality."
        
        # Tokenize
        inputs = tokenizer(text, return_tensors="pt")
        print(f"✓ Input text tokenized: {len(inputs['input_ids'][0])} tokens")
        
        # Run inference
        with torch.no_grad():
            outputs = model(**inputs)
        
        print(f"✓ Output shape: {outputs.last_hidden_state.shape}")
        print("✓ Inference successful")
    except Exception as e:
        print(f"✗ Inference failed: {str(e)}")
        sys.exit(1)

def test_tokenizer_functionality(tokenizer):
    """Test tokenizer features"""
    print_section("5. Testing Tokenizer Functionality")
    
    try:
        # Test text
        text = "HuggingFace is awesome!"
        
        # Basic tokenization
        tokens = tokenizer.tokenize(text)
        print(f"✓ Tokenization: {tokens}")
        
        # Encoding
        encoded = tokenizer.encode(text)
        print(f"✓ Encoding: {encoded}")
        
        # Decoding
        decoded = tokenizer.decode(encoded)
        print(f"✓ Decoding: {decoded}")
        
    except Exception as e:
        print(f"✗ Tokenizer testing failed: {str(e)}")
        sys.exit(1)

def main():
    """Main function to run all tests"""
    print("\nHuggingFace Setup Test Script")
    print("Time:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    
    # Run all tests
    test_installation()
    test_authentication()
    model, tokenizer = test_model_loading()
    test_inference(model, tokenizer)
    test_tokenizer_functionality(tokenizer)
    
    print("\n✓ All tests completed successfully!")

if __name__ == "__main__":
    main() 