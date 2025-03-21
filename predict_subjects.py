import torch
from transformers import AutoTokenizer
from src.models.model import SubjectClassifier
import yaml
import argparse

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def predict_subjects(abstract, model, tokenizer, config):
    # Prepare input
    inputs = tokenizer(
        abstract,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt"
    ).to(model.device)
    
    # Get predictions
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.sigmoid(outputs.logits)
        
    # Convert to binary predictions
    predictions = (predictions > 0.5).squeeze()
    
    # Get predicted subjects
    predicted_subjects = [
        subject for i, subject in enumerate(config['label_encoder']['classes'])
        if predictions[i].item()
    ]
    
    return predicted_subjects

def main():
    parser = argparse.ArgumentParser(description='Predict subjects from an abstract')
    parser.add_argument('--config', default='configs/config.yaml', help='Path to config file')
    parser.add_argument('--model_path', default='outputs/model', help='Path to trained model')
    parser.add_argument('--abstract', type=str, required=True, help='Abstract text to classify')
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Load model and tokenizer
    model = SubjectClassifier(config)
    model.load_state_dict(torch.load(f"{args.model_path}/pytorch_model.bin"))
    model.eval()
    
    tokenizer = AutoTokenizer.from_pretrained(config['model_name'])
    
    # Predict subjects
    predicted_subjects = predict_subjects(args.abstract, model, tokenizer, config)
    
    # Print results
    print("\nAbstract:")
    print("-" * 50)
    print(args.abstract)
    print("\nPredicted Subjects:")
    print("-" * 50)
    for subject in predicted_subjects:
        print(f"- {subject}")

if __name__ == "__main__":
    main() 