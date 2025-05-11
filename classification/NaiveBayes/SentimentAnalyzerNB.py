import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from tokenizer.tokenizer import Tokenizer
import pickle
import os

DATA_SET = "./data_set/reviews.pkl"
MODEL_PATH = "./models/sentiment_model_nb.pkl"


class SentimentAnalyzerNB:
    def __init__(self, data_path, threshold=3.0):
        """
        Initialize the sentiment analyzer with dataset path and rating threshold

        Args:
            data_path (str): Path to the JSON review data
            threshold (float): Star rating threshold for positive/negative classification
        """
        self.data_path = data_path
        self.threshold = threshold
        self.tokenizer = Tokenizer()
        self.model = None
        self.vectorizer = None

    def load_data(self):
        """Load the review data from JSON file or JSON Lines format"""
        df = pd.read_pickle(self.data_path)
        print(f"Loaded {len(df)} reviews")
        return df

    def prepare_data(self, df):
        """Prepare the data for sentiment analysis"""
        # Clean the text data
        df["cleaned_text"] = df["text"].apply(
            lambda x: " ".join(self.tokenizer.get_tokens(x))
        )

        # Convert stars to sentiment labels
        df["sentiment"] = df["stars"].apply(lambda x: 1 if x >= self.threshold else 0)

        # Display class distribution
        print(f"Positive reviews: {df['sentiment'].sum()}")
        print(f"Negative reviews: {len(df) - df['sentiment'].sum()}")

        return df

    def train_model(self, df, test_size=0.2, random_state=42):
        """Train the MultinomialNB model"""
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            df["cleaned_text"],
            df["sentiment"],
            test_size=test_size,
            random_state=random_state,
            stratify=df["sentiment"],
        )

        # Create a pipeline with TF-IDF and MultinomialNB
        self.model = Pipeline(
            [
                (
                    "tfidf",
                    TfidfVectorizer(max_features=5000, ngram_range=(1, 2)),
                ),
                ("classifier", MultinomialNB(alpha=0.1)),
            ]
        )

        # Train the model
        self.model.fit(X_train, y_train)

        # Save the model
        with open(MODEL_PATH, "wb") as f:
            pickle.dump(self.model, f)

        # Evaluate the model
        y_pred = self.model.predict(X_test)

        # Print metrics
        print("\nModel Evaluation:")
        print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
        print("\nClassification Report:")
        print(
            classification_report(y_test, y_pred, target_names=["Negative", "Positive"])
        )

        # Create confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["Negative", "Positive"],
            yticklabels=["Negative", "Positive"],
        )
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("Confusion Matrix")
        plt.tight_layout()
        plt.savefig("confusion_matrix.png")

        return

    def analyze_feature_importance(self, n_features=20):
        """Analyze the most important features for each class"""
        # Get the vectorizer and classifier from the pipeline
        vectorizer = self.model.named_steps["tfidf"]
        classifier = self.model.named_steps["classifier"]

        # Get feature names
        feature_names = vectorizer.get_feature_names_out()

        # Get the log probabilities for each class
        pos_probs = classifier.feature_log_prob_[1]  # Positive class
        neg_probs = classifier.feature_log_prob_[0]  # Negative class

        # Create DataFrames for visualization
        pos_features = (
            pd.DataFrame(
                {
                    "Feature": feature_names,
                    "Log Probability": pos_probs,
                    "Sentiment": "Positive",
                }
            )
            .sort_values("Log Probability", ascending=False)
            .head(n_features)
        )

        neg_features = (
            pd.DataFrame(
                {
                    "Feature": feature_names,
                    "Log Probability": neg_probs,
                    "Sentiment": "Negative",
                }
            )
            .sort_values("Log Probability", ascending=False)
            .head(n_features)
        )

        # Combine features for visualization
        all_features = pd.concat([pos_features, neg_features])

        # Plot the most important features
        plt.figure(figsize=(16, 8))
        sns.barplot(
            x="Log Probability",
            y="Feature",
            data=all_features,
            palette="viridis",
            hue="Sentiment",
        )
        plt.title("Top Features by Sentiment")
        plt.tight_layout()
        plt.savefig("feature_importance.png")

    def predict_sentiment(self, reviews):
        """Predict sentiment for new reviews"""
        if isinstance(reviews, str):
            reviews = [reviews]

        # Preprocess the reviews
        preprocessed_reviews = [
            " ".join(self.tokenizer.get_tokens(review)) for review in reviews
        ]

        # Make predictions
        predictions = self.model.predict(preprocessed_reviews)
        probabilities = self.model.predict_proba(preprocessed_reviews)

        results = []
        for i, review in enumerate(reviews):
            sentiment = "Positive" if predictions[i] == 1 else "Negative"
            confidence = probabilities[i][predictions[i]]
            results.append(
                {"review": review, "sentiment": sentiment, "confidence": confidence}
            )

        return results

    def init(self):
        # Check if model already exists
        if os.path.exists(MODEL_PATH):
            print(f"Loading existing model from {MODEL_PATH}")
            with open(MODEL_PATH, "rb") as f:
                self.model = pickle.load(f)
            return
        else:
            # Load and prepare data
            df = self.load_data()
            prepared_df = self.prepare_data(df)
            # Train and evaluate the model
            self.train_model(prepared_df)


# Example usage
if __name__ == "__main__":
    # Initialize the analyzer
    analyzer = SentimentAnalyzerNB(DATA_SET, threshold=3.5)

    # Load data and train model
    analyzer.init()

    # UTILS: Analyze feature importance for stats
    # analyzer.analyze_feature_importance()

    # Example with manual predictions
    test_reviews = [
        "The food was amazing and the service was excellent!",
        "Worst experience ever. The staff was rude and the food was cold.",
        "The ambiance was nice but the food was just okay.",
        "Never add a greater dinner in my life. Go !!!",
    ]

    results = analyzer.predict_sentiment(test_reviews)

    print("\nExample Predictions:")
    for result in results:
        print(f"Review: {result['review']}")
        print(
            f"-> Sentiment: {result['sentiment']} (Confidence: {result['confidence']:.4f})"
        )

"""
WITH NON PREPROCESSED DATASET
Model Evaluation:
Accuracy: 0.8874

Classification Report:
              precision    recall  f1-score   support

    Negative       0.90      0.73      0.81       686
    Positive       0.88      0.96      0.92      1454

    accuracy                           0.89      2140
   macro avg       0.89      0.85      0.86      2140
weighted avg       0.89      0.89      0.88      2140


Example Predictions:
Review: The food was amazing and the service was excellent!
-> Sentiment: Positive (Confidence: 0.9894)
Review: Worst experience ever. The staff was rude and the food was cold.
-> Sentiment: Negative (Confidence: 0.9959)
Review: The ambiance was nice but the food was just okay.
-> Sentiment: Negative (Confidence: 0.8075)
Review: Never add a greater dinner in my life. Go !!!
-> Sentiment: Positive (Confidence: 0.7062)
"""

"""
"""
