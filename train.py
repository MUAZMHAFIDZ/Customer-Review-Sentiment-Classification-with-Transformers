from datasets import Dataset, Value
import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import pandas as pd

local_model_path = "./local_model"

# Load data
df = pd.read_csv("data.csv")
train_df = df.sample(frac=0.8, random_state=42)
test_df = df.drop(train_df.index)

# Load tokenizer
if os.path.exists(local_model_path):
    tokenizer = AutoTokenizer.from_pretrained(local_model_path)
else:
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

def tokenize(batch):
    return tokenizer(batch["text"], truncation=True, padding=True)

train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)

train_dataset = train_dataset.map(tokenize, batched=True)
test_dataset = test_dataset.map(tokenize, batched=True)

train_dataset = train_dataset.rename_column("label", "labels").cast_column("labels", Value("int64"))
test_dataset = test_dataset.rename_column("label", "labels").cast_column("labels", Value("int64"))

train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
test_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

# Load model
if os.path.exists(local_model_path):
    model = AutoModelForSequenceClassification.from_pretrained(local_model_path)
else:
    model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)

# Training args (tanpa evaluation_strategy)
training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    save_total_limit=2,
    logging_dir="./logs",
    logging_steps=10,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,  # masih bisa disimpan meskipun tidak otomatis dievaluasi
)

trainer.train()

# Simpan model dan tokenizer
save_path = "./my_test_model"
model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)
print(f"Model saved to {save_path}")
