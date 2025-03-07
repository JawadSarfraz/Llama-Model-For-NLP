try:
    import llama  # Attempt to import the llama package
    print("LLaMA package is successfully installed!")
except ImportError:
    print("LLaMA package is NOT installed.")
except Exception as e:
    print(f"An error occurred: {e}")