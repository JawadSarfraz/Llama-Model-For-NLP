from transformers import Trainer, TrainingArguments
from transformers import LlamaForSequenceClassification, LlamaTokenizer
from huggingface_hub import login
from datasets import load_dataset

# Log into Hugging Face
login("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")

# Load the tokenizer and model
model_name = "huggyllama/llama-7b"
tokenizer = LlamaTokenizer.from_pretrained(model_name)
model = LlamaForSequenceClassification.from_pretrained(
    model_name,
    load_in_8bit=True,  # Use 8-bit quantization
    device_map="auto",  # Automatically assign model to GPU
    num_labels=3        # Adjust num_labels based on classification needs
)

# Load dataset from a JSON file
dataset = load_dataset("json", data_files="sample_data.json")

# Filter out invalid examples (ensure "abstract" exists and is not None)
dataset = dataset.filter(lambda example: "abstract" in example and example["abstract"] is not None)

# Define label mapping
label_mapping = {
    "330": 0,  # Example: Economics
    "F41": 1,  # Example: Trade
    "G21": 2   # Example: Banking/Finance
}

# Add labels to the dataset
def add_labels(example):
    example["label"] = label_mapping.get(example["classification_ddc"][0], -1)  # Default -1 for unknown
    return example

dataset = dataset.map(add_labels)

# Filter out samples with invalid labels (-1)
dataset = dataset.filter(lambda example: example["label"] != -1)

# Tokenize the dataset
def tokenize_function(example):
    if "abstract" in example and example["abstract"]:
        abstract_text = " ".join(example["abstract"])
    else:
        abstract_text = ""
    
    return tokenizer(
        abstract_text,
        truncation=True,
        padding="max_length",
        max_length=512,
    )

# Split the dataset into train and test sets
train_test_split = dataset["train"].train_test_split(test_size=0.2)
train_dataset = train_test_split["train"].map(tokenize_function, batched=True)
test_dataset = train_test_split["test"].map(tokenize_function, batched=True)

# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
)

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
)

# Fine-tune the model
trainer.train()

# Evaluate the model
results = trainer.evaluate()
print("Evaluation Results:", results)

# Example text for prediction
new_article = "The German banking system faces significant integration challenges."

# Tokenize the new article
inputs = tokenizer(new_article, return_tensors="pt", truncation=True, padding="max_length", max_length=512)

# Make prediction
outputs = model(**inputs)
predicted_label = outputs.logits.argmax(dim=1).item()

# Map the predicted label to the category
label_map = {0: "Economics", 1: "Trade", 2: "Banking/Finance"}
print("Predicted Category:", label_map[predicted_label])
