from transformers import Trainer, TrainingArguments
from transformers import LlamaForSequenceClassification, LlamaTokenizer
from huggingface_hub import login
from datasets import load_dataset
import pyarrow as pa

# Log into Hugging Face
login("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")

# Load the tokenizer and model
model_name = "huggyllama/llama-7b"
tokenizer = LlamaTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token  # Use eos_token as padding token

model = LlamaForSequenceClassification.from_pretrained(
    model_name,
    load_in_8bit=True,
    device_map="auto",
    num_labels=3
)

# Load dataset and remove pyarrow schema
dataset = load_dataset("json", data_files="sample_data.json").with_format("python")

# Filter out invalid examples
dataset = dataset.filter(lambda example: "abstract" in example and example["abstract"] is not None)

# Map labels
label_mapping = {"330": 0, "F41": 1, "G21": 2}

def add_labels(example):
    example["label"] = label_mapping.get(example["classification_ddc"][0], -1)
    return example

dataset = dataset.map(add_labels)
dataset = dataset.filter(lambda example: example["label"] != -1)

# Fixed max length
MAX_LENGTH = 80

# Tokenize dataset
def tokenize_function(example):
    abstract_text = " ".join([str(item) for item in example["abstract"]]) if "abstract" in example and example["abstract"] else ""
    return tokenizer(
        abstract_text,
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH  # Use fixed max length
    )

# Tokenize and split the dataset
train_test_split = dataset["train"].train_test_split(test_size=0.2)
train_dataset = train_test_split["train"].map(tokenize_function, batched=True)
test_dataset = train_test_split["test"].map(tokenize_function, batched=True)

# Dynamically reset schema for the tokenized dataset
def reset_schema(dataset, max_length):
    def adjust_length(batch):
        batch["input_ids"] = [
            ids[:max_length] + [tokenizer.pad_token_id] * (max_length - len(ids))
            for ids in batch["input_ids"]
        ]
        batch["attention_mask"] = [
            mask[:max_length] + [0] * (max_length - len(mask))
            for mask in batch["attention_mask"]
        ]
        return batch
    return dataset.map(adjust_length, batched=True)

train_dataset = reset_schema(train_dataset, MAX_LENGTH)
test_dataset = reset_schema(test_dataset, MAX_LENGTH)

# Cast schema manually (optional)
def cast_dataset(dataset, max_length):
    schema = pa.schema([
        ("input_ids", pa.list_(pa.int32(), max_length)),  # Define `input_ids` with fixed length
        ("attention_mask", pa.list_(pa.int32(), max_length)),  # Define `attention_mask` with fixed length
        ("label", pa.int32())  # Label column
    ])
    return dataset.cast(schema)

train_dataset = cast_dataset(train_dataset, MAX_LENGTH)
test_dataset = cast_dataset(test_dataset, MAX_LENGTH)

# Set dataset format to torch
train_dataset.set_format("torch")
test_dataset.set_format("torch")

# Training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
)

# Trainer setup
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
)

# Train and evaluate
trainer.train()
results = trainer.evaluate()
print("Evaluation Results:", results)

# Example text for prediction
new_article = "The German banking system faces significant integration challenges."

# Tokenize the input with the same max_length
inputs = tokenizer(new_article, return_tensors="pt", truncation=True, padding="max_length", max_length=MAX_LENGTH)

# Make prediction
outputs = model(**inputs)
predicted_label = outputs.logits.argmax(dim=1).item()

# Map label to category
label_map = {0: "Economics", 1: "Trade", 2: "Banking/Finance"}
print("Predicted Category:", label_map[predicted_label])
