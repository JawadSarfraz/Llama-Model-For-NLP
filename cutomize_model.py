from transformers import Trainer, TrainingArguments
from transformers import LlamaForSequenceClassification, LlamaTokenizer
from huggingface_hub import login

login("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")


# Load the tokenizer and model
model_name = "huggyllama/llama-7b" 
tokenizer = LlamaTokenizer.from_pretrained(model_name)
model = LlamaForSequenceClassification.from_pretrained(model_name, num_labels=3)  # 3 labels for our categories

from datasets import load_dataset

# Load the dataset from a CSV file
dataset = load_dataset("csv", data_files="news_articles.csv")

# Split into training and testing sets
train_dataset = dataset["train"].train_test_split(test_size=0.2)["train"]
test_dataset = dataset["train"].train_test_split(test_size=0.2)["test"]

# Tokenize the text
def tokenize_function(example):
    return tokenizer(example["text"], truncation=True, padding="max_length", max_length=512)

train_dataset = train_dataset.map(tokenize_function, batched=True)
test_dataset = test_dataset.map(tokenize_function, batched=True)

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
new_article = "The government passed a new law that affects international relations."

# Tokenize the new article
inputs = tokenizer(new_article, return_tensors="pt", truncation=True, padding="max_length", max_length=512)

# Make prediction
outputs = model(**inputs)
predicted_label = outputs.logits.argmax(dim=1).item()

# Map the predicted label to the category
label_map = {0: "Politics", 1: "Sports", 2: "Technology"}
print("Predicted Category:", label_map[predicted_label])
