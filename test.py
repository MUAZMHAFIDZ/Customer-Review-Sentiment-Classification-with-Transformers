from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F

model_path = "./my_test_model"

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

inputs = tokenizer("I love learning NLP!", return_tensors="pt")
outputs = model(**inputs)

probs = F.softmax(outputs.logits, dim=1)
label = torch.argmax(probs).item()

labels_map = ["negative", "positive"]
print("Sentiment:", labels_map[label])
