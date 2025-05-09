import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer
from datasets import Dataset
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Chargement des données
df = pd.read_pickle('../../data_set/reviews.pkl')

# Conversion des étoiles en classes (négatif, neutre, positif)
def convert_stars_to_sentiment(stars):
    if stars <= 2:
        return 0  # Négatif
    elif stars == 3:
        return 1  # Neutre
    else:
        return 2  # Positif

df['labels'] = df['stars'].apply(convert_stars_to_sentiment)

# Division des données
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['labels'])

# Conversion en datasets Hugging Face
train_dataset = Dataset.from_pandas(train_df[['text', 'labels']])
test_dataset = Dataset.from_pandas(test_df[['text', 'labels']])

# Chargement du tokenizer et du modèle
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)

# Tokenization des données
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)

tokenized_train = train_dataset.map(tokenize_function, batched=True)
tokenized_test = test_dataset.map(tokenize_function, batched=True)

# Fonction de métrique d'évaluation
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

# Configuration de l'entraînement
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
)

# Création du trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_test,
    compute_metrics=compute_metrics,
)

# Entraînement du modèle
trainer.train()

# Évaluation du modèle
eval_results = trainer.evaluate()
print(f"Résultats d'évaluation: {eval_results}")

# Sauvegarder le modèle
model_path = "./yelp_sentiment_model"
model.save_pretrained(model_path)
tokenizer.save_pretrained(model_path)

# Exemple d'utilisation
def predict_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
    
    prediction = torch.argmax(outputs.logits, dim=-1).item()
    sentiment_map = {0: "Négatif", 1: "Neutre", 2: "Positif"}
    return sentiment_map[prediction]

# Test avec quelques exemples
test_reviews = [
    "Le service était horrible et la nourriture froide.",
    "Le restaurant était correct, rien d'extraordinaire.",
    "Une expérience incroyable ! La meilleure nourriture que j'ai jamais goûtée."
]

for review in test_reviews:
    print(f"Review: {review}")
    print(f"Sentiment: {predict_sentiment(review)}\n")
