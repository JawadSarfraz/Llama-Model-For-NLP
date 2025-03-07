from transformers import Trainer, TrainingArguments
from transformers import LlamaForSequenceClassification, LlamaTokenizer
from huggingface_hub import login
from datasets import load_dataset
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer

# Log into Hugging Face
login("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")

# Load the tokenizer and model
model_name = "huggyllama/llama-7b"
tokenizer = LlamaTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token  # Use eos_token as padding token

# Load dataset and remove pyarrow schema
dataset = load_dataset("json", data_files="data/sample_data.json").with_format("python")

# Filter out invalid examples
dataset = dataset.filter(lambda example: "abstract" in example and example["abstract"] is not None)

# Create subject vocabulary
all_subjects = set()
for example in dataset["train"]:
    if "subject" in example and example["subject"]:
        all_subjects.update(example["subject"])

# Convert subjects to list and create label encoder
subject_list = sorted(list(all_subjects))
label_encoder = MultiLabelBinarizer()
label_encoder.fit([subject_list])  # Fit on all possible subjects

# Function to encode labels
def encode_labels(example):
    if "subject" in example and example["subject"]:
        example["labels"] = label_encoder.transform([example["subject"]])[0]
    else:
        example["labels"] = np.zeros(len(subject_list))
    return example

# Apply label encoding
dataset = dataset.map(encode_labels)

# Fixed max length
MAX_LENGTH = 80

# Tokenize dataset
def tokenize_function(example):
    abstract_text = " ".join([str(item) for item in example["abstract"]]) if "abstract" in example and example["abstract"] else ""
    return tokenizer(
        abstract_text,
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH
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

# Cast schema manually
import pyarrow as pa

def cast_dataset(dataset, max_length):
    schema = pa.schema([
        ("input_ids", pa.list_(pa.int32(), max_length)),
        ("attention_mask", pa.list_(pa.int32(), max_length)),
        ("labels", pa.list_(pa.int32(), len(subject_list)))
    ])
    return dataset.cast(schema)

train_dataset = cast_dataset(train_dataset, MAX_LENGTH)
test_dataset = cast_dataset(test_dataset, MAX_LENGTH)

# Set dataset format to torch
train_dataset.set_format("torch")
test_dataset.set_format("torch")

# Initialize model with number of labels equal to number of subjects
model = LlamaForSequenceClassification.from_pretrained(
    model_name,
    load_in_8bit=True,
    device_map="auto",
    num_labels=len(subject_list),
    problem_type="multi_label_classification"
)

# Training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    metric_for_best_model="f1",
    load_best_model_at_end=True,
)

# Custom compute metrics function for multi-label classification
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = (predictions > 0).astype(int)
    
    # Calculate metrics
    from sklearn.metrics import f1_score, precision_score, recall_score
    f1 = f1_score(labels, predictions, average='weighted')
    precision = precision_score(labels, predictions, average='weighted')
    recall = recall_score(labels, predictions, average='weighted')
    
    return {
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

# Trainer setup
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics,
)

# Train and evaluate
trainer.train()
results = trainer.evaluate()
print("Evaluation Results:", results)

# Example text for prediction
new_article = "The German banking system faces significant integration challenges."
inputs = tokenizer(new_article, return_tensors="pt", truncation=True, padding="max_length", max_length=MAX_LENGTH)
outputs = model(**inputs)
predictions = (outputs.logits > 0).squeeze().numpy()

# Get predicted subjects
predicted_subjects = label_encoder.inverse_transform([predictions])
print("\nPredicted Subjects:", predicted_subjects[0])
