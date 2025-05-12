import pandas as pd
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report

class YelpReviewClassifier:
    def __init__(self, model_path='./models/logistic_regression_model.pkl'):
        self.model_path = model_path
        self.pipeline = None

    def train(self, data_path):
        # Load dataset
        df = pd.read_json(data_path, lines=True)
        df['label'] = df['stars'].apply(lambda x: 1 if x > 3.5 else 0)

        X = df['text']
        y = df['label']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Define pipeline (vectorizer + model)
        self.pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(stop_words='english', max_features=5000)),
            ('clf', LogisticRegression())
        ])

        # Train
        self.pipeline.fit(X_train, y_train)

        # Evaluate
        y_pred = self.pipeline.predict(X_test)
        print(f"Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
        print("Classification Report:")
        print(classification_report(y_test, y_pred))

        # Save pipeline
        self.save()

    def save(self):
        with open(self.model_path, 'wb') as f:
            pickle.dump(self.pipeline, f)

    def load(self):
        if os.path.exists(self.model_path):
            with open(self.model_path, 'rb') as f:
                self.pipeline = pickle.load(f)
        else:
            raise FileNotFoundError("Model file not found.")

    def predict_stars(self, reviews):
        if not self.pipeline:
            raise ValueError("Pipeline not loaded.")
        return self.pipeline.predict(reviews)

# Exemple d'utilisation
if __name__ == "__main__":
    classifier = YelpReviewClassifier()

    # Première fois (entraînement et sauvegarde)
    #classifier.train('data_set/yelp_subset_review.json')

    # Utilisation ultérieure
    classifier.load()
    reviews = [
        "Never had a greater dinner in my life. Go !!!",
        "The food was amazing and the service was excellent!",
        "The service was good but the food was cold."
    ]
    predictions = classifier.predict_stars(reviews)
    for review, pred in zip(reviews, predictions):
        print(f"Review: {review} --> Predicted label: {pred} (1 = Positive, 0 = Negative)")
