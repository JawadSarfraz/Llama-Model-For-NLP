import os
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import huggingface_hub
from huggingface_hub import HfFolder
import glob

def check_llama_installation():
    """Check if LLaMA-7B is installed in the environment and locate its path."""
    print("Checking LLaMA-7B installation...")
    
    # Check if we're in a virtual environment
    in_venv = sys.prefix != sys.base_prefix
    print(f"Running in virtual environment: {in_venv}")
    if in_venv:
        print(f"Virtual environment path: {sys.prefix}")
    
    # Check if Hugging Face token is set
    token = HfFolder.get_token()
    if token:
        print("Hugging Face token is set")
    else:
        print("Hugging Face token is NOT set")
    
    # Check for LLaMA model in cache
    cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
    print(f"Checking Hugging Face cache at: {cache_dir}")
    
    # Look for LLaMA model files
    llama_paths = []
    for root, dirs, files in os.walk(cache_dir):
        for file in files:
            if "llama" in file.lower() or "llama" in root.lower():
                llama_paths.append(os.path.join(root, file))
    
    if llama_paths:
        print(f"Found {len(llama_paths)} potential LLaMA files in cache:")
        for path in llama_paths[:5]:  # Show first 5 paths
            print(f"  - {path}")
        if len(llama_paths) > 5:
            print(f"  - ... and {len(llama_paths) - 5} more")
    else:
        print("No LLaMA files found in cache")
    
    # Try to load the model directly
    try:
        print("\nTrying to load LLaMA-7B model directly...")
        model_name = "huggyllama/llama-7b"  # Using the open source version
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        print(f"Successfully loaded tokenizer from {model_name}")
        
        # Try to load a small part of the model to check
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            load_in_4bit=True,
            device_map="auto",
            torch_dtype=torch.float16,
            max_memory={0: "2GB"}  # Limit memory usage for testing
        )
        print(f"Successfully loaded model from {model_name}")
        return True
    except Exception as e:
        print(f"Error loading model directly: {e}")
        
        # Try to find local model files
        print("\nTrying to find local model files...")
        local_paths = [
            os.path.join(sys.prefix, "lib", "python3.10", "site-packages", "llama"),
            os.path.join(sys.prefix, "lib", "python3.10", "site-packages", "llama2"),
            os.path.join(sys.prefix, "lib", "python3.10", "site-packages", "llama-2"),
            os.path.join(os.path.expanduser("~"), "llama"),
            os.path.join(os.path.expanduser("~"), "llama2"),
            os.path.join(os.path.expanduser("~"), "llama-2"),
            os.path.join(os.path.expanduser("~"), "models", "llama"),
            os.path.join(os.path.expanduser("~"), "models", "llama2"),
            os.path.join(os.path.expanduser("~"), "models", "llama-2"),
        ]
        
        for path in local_paths:
            if os.path.exists(path):
                print(f"Found potential LLaMA directory: {path}")
                files = os.listdir(path)
                print(f"  Files: {files[:5]}")
                if len(files) > 5:
                    print(f"  ... and {len(files) - 5} more")
        
        return False

if __name__ == "__main__":
    check_llama_installation() 