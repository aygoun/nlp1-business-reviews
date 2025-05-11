import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer
from datasets import Dataset
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

class ClassificationTransformer:
    def __init__(self, model_path=None):
        self.model_name = "distilbert-base-uncased"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name, num_labels=3)

        if model_path:
            self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)

    def convert_stars_to_sentiment(self, stars):
        if stars <= 2:
            return 0  # Négatif
        elif stars == 3:
            return 1  # Neutre
        else:
            return 2  # Positif

    def load_and_prepare_data(self, data_path):
        df = pd.read_pickle(data_path)
        df['labels'] = df['stars'].apply(self.convert_stars_to_sentiment)
        train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['labels'])
        train_dataset = Dataset.from_pandas(train_df[['text', 'labels']])
        test_dataset = Dataset.from_pandas(test_df[['text', 'labels']])
        return train_dataset, test_dataset

    def tokenize_function(self, examples):
        return self.tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)

    def compute_metrics(self, pred):
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

    def train(self, train_dataset, test_dataset):
        tokenized_train = train_dataset.map(self.tokenize_function, batched=True)
        tokenized_test = test_dataset.map(self.tokenize_function, batched=True)

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

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_train,
            eval_dataset=tokenized_test,
            compute_metrics=self.compute_metrics,
        )

        trainer.train()
        eval_results = trainer.evaluate()
        print(f"Résultats d'évaluation: {eval_results}")

        model_path = "./yelp_sentiment_model"
        self.model.save_pretrained(model_path)
        self.tokenizer.save_pretrained(model_path)

        return model_path

    def predict_sentiment(self, text):
        device = next(self.model.parameters()).device
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)

        prediction = torch.argmax(outputs.logits, dim=-1).item()
        star_rating = prediction + 1

        return star_rating

# Initialize with a saved model
#model = ClassificationTransformer(model_path="./model_save")

# Test with some examples
#test_reviews = [
#    "The service was terrible and the food was cold.",
#    "The restaurant was okay, nothing extraordinary.",
#    "An incredible experience! The best food I've ever tasted.",
#    "The food was delicious but the staff was mean. I will not come back."
#]

#for review in test_reviews:
#    print(f"Review: {review}")
#    print(f"Sentiment: {model.predict_sentiment(review)}\n")
