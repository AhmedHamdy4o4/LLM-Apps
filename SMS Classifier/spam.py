import os
import torch
from datasets import load_dataset, DatasetDict, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import evaluate
import numpy as np

# 1. Load Dataset
dataset = load_dataset("sms_spam")
dataset = dataset.rename_column("label", "labels")

# 2. Load Tokenizer and Model
model_name = "distilbert-base-uncased" 
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

# Tokenization function for the dataset
def tokenize_function(examples):
    return tokenizer(examples["sms"], padding="max_length", truncation=True)

# Split the dataset
split_datasets = dataset["train"].train_test_split(test_size=0.2, seed=42)
train_dataset = split_datasets['train']
eval_dataset = split_datasets['test']

# Apply the tokenization function to the datasets
tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True)
tokenized_eval_dataset = eval_dataset.map(tokenize_function, batched=True)

# Define a function to compute metrics
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    
    f1_metric = evaluate.load("f1")
    accuracy_metric = evaluate.load("accuracy")
    
    f1_score = f1_metric.compute(predictions=predictions, references=labels, average="weighted")
    accuracy_score = accuracy_metric.compute(predictions=predictions, references=labels)
    
    return {"accuracy": accuracy_score["accuracy"], "f1": f1_score["f1"]}

# Training arguments object
training_args = TrainingArguments(
    output_dir=os.getcwd(),
    eval_strategy="epoch", 
    num_train_epochs=3, 
    per_device_train_batch_size=16, 
    per_device_eval_batch_size=16,
    learning_rate=2e-5,  
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=100,
    report_to="none" 
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_eval_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

# Perform training
trainer.train()

# Perform evaluation
results = trainer.evaluate()
print(f"Model evaluation results: {results}")

# 3. Test the model with new data
test_data = [
    "Hey, are we still meeting at 6 pm for dinner?",
    "Congratulations!.You’ve won a $1,000 Walmart Gift Card. Click here to claim: http://bit.ly/xxxx",
    "URGENT! Your account has been suspended. Verify your details immediately at http://secure-login-fakebank.com"
]

# test  model
encodings = tokenizer(test_data, padding=True, truncation=True, return_tensors="pt")
with torch.no_grad():
    outputs = model(**encodings)
predictions = torch.argmax(outputs.logits, dim=1)
mapper = {0: "ham", 1: "spam"}
for i, label_id in enumerate(predictions):
    class_pred = mapper[label_id.item()]
    print(f"Input text {i+1}: {test_data[i]}")
    print(f"Predicted class: {class_pred}\n")