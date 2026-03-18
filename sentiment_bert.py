import numpy as np
from datasets import load_dataset
from transformers import (
    DistilBertTokenizer,
    DistilBertForSequenceClassification,
    TrainingArguments,
    Trainer
)
from sklearn.metrics import accuracy_score, classification_report

# ── 1. Load full IMDB dataset ──────────────────────────────────────────────────
dataset = load_dataset("imdb")
train_dataset = dataset["train"]   # 25,000 reviews
test_dataset  = dataset["test"]    # 25,000 reviews

# ── 2. Tokenize ────────────────────────────────────────────────────────────────
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

def tokenize(batch):
    return tokenizer(
        batch["text"],
        padding="max_length",
        truncation=True,
        max_length=512
    )

train_dataset = train_dataset.map(tokenize, batched=True)
test_dataset  = test_dataset.map(tokenize,  batched=True)

train_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])
test_dataset.set_format("torch",  columns=["input_ids", "attention_mask", "label"])

# ── 3. Load model ──────────────────────────────────────────────────────────────
model = DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=2
)

# ── 4. Metrics ─────────────────────────────────────────────────────────────────
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    return {"accuracy": accuracy_score(labels, preds)}

# ── 5. Training arguments ──────────────────────────────────────────────────────
training_args = TrainingArguments(
    output_dir="./distilbert-imdb-full",
    num_train_epochs=3,
    per_device_train_batch_size=128,
    per_device_eval_batch_size=128,
    save_strategy="steps",
    save_steps=1000,
    fp16=True,
    seed=42
)

# ── 6. Train ───────────────────────────────────────────────────────────────────
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics
)

trainer.train()

# ── 7. Evaluate ────────────────────────────────────────────────────────────────
results = trainer.evaluate()
print(f"\nTest Accuracy: {results['eval_accuracy']:.4f}")

# ── 8. Classification report ───────────────────────────────────────────────────
predictions = trainer.predict(test_dataset)
y_pred = np.argmax(predictions.predictions, axis=1)
y_true = predictions.label_ids

print(classification_report(y_true, y_pred, target_names=["Negative", "Positive"]))