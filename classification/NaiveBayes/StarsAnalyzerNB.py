import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from tokenizer.tokenizer import Tokenizer
import time
import pickle
import os

DATA_SET = "./data_set/reviews.pkl"
MODEL_PATH = "./models/stars_model_nb.pkl"


class SentimentAnalyzerNB:
    def __init__(self, data_path):
        """
        Initialize the sentiment analyzer with dataset path

        Args:
            data_path (str): Path to the JSON review data
        """
        self.data_path = data_path
        self.tokenizer = Tokenizer()
        self.model = None
        self.vectorizer = None

    def load_data(self):
        """Load the review data from JSON file or JSON Lines format"""
        df = pd.read_pickle(self.data_path)

        print(f"\nData Loading Summary:")
        print(f"Total reviews loaded: {len(df)}")
        print(f"Columns available: {df.columns.tolist()}")
        print(f"Sample of star ratings distribution:")
        print(df["stars"].value_counts().sort_index())
        return df

    def prepare_data(self, df):
        """Prepare the data for star rating prediction"""
        print("\nData Preparation Summary:")
        print(f"Original text samples: {len(df)}")

        # Clean the text data
        df["cleaned_text"] = df["text"].apply(
            lambda x: " ".join(self.tokenizer.get_tokens(x))
        )

        # Check for empty texts after cleaning
        empty_texts = df["cleaned_text"].str.strip().eq("").sum()
        if empty_texts > 0:
            print(f"Warning: {empty_texts} reviews became empty after cleaning")
            df = df[df["cleaned_text"].str.strip().ne("")]

        print(f"Final dataset size: {len(df)} reviews")
        print("\nStar Rating Distribution:")
        star_counts = df["stars"].value_counts().sort_index()
        for star, count in star_counts.items():
            print(f"{star} stars: {count} reviews ({count/len(df)*100:.1f}%)")

        return df

    def train_model(self, df, test_size=0.2, random_state=42):
        """Train the MultinomialNB model for star rating prediction"""
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            df["cleaned_text"],
            df["stars"],
            test_size=test_size,
            random_state=random_state,
            stratify=df["stars"],
        )

        print(f"\nTraining Data Summary:")
        print(f"Training samples: {len(X_train)}")
        print(f"Testing samples: {len(X_test)}")
        print("\nTraining set star distribution:")
        print(y_train.value_counts().sort_index())

        # Create a pipeline with TF-IDF and MultinomialNB
        self.model = Pipeline(
            [
                ("tfidf", TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
                ("classifier", MultinomialNB(alpha=0.1)),
            ]
        )

        # Train the model
        print("\nTraining model...")
        start_time = time.time()
        self.model.fit(X_train, y_train)
        end_time = time.time()
        print(f"Training completed in {end_time - start_time:.2f} seconds.")

        # Save the model
        with open(MODEL_PATH, "wb") as f:
            pickle.dump(self.model, f)

        # Print vocabulary size
        vectorizer = self.model.named_steps["tfidf"]
        print(f"Vocabulary size: {len(vectorizer.get_feature_names_out())}")

        # Evaluate the model
        y_pred = self.model.predict(X_test)

        # Print metrics
        print("\nModel Evaluation:")
        print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))

        # Create confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=range(1, 6),
            yticklabels=range(1, 6),
        )
        plt.xlabel("Predicted Star Rating")
        plt.ylabel("Actual Star Rating")
        plt.title("Confusion Matrix")
        plt.tight_layout()
        plt.savefig("confusion_matrix.png")

        return

    def analyze_feature_importance(self, n_features=20):
        """Analyze the most important features for each star rating"""
        # Get the vectorizer and classifier from the pipeline
        vectorizer = self.model.named_steps["tfidf"]
        classifier = self.model.named_steps["classifier"]

        # Get feature names
        feature_names = vectorizer.get_feature_names_out()

        # Create a figure with subplots for each star rating
        plt.figure(figsize=(20, 15))

        for star in range(1, 6):
            plt.subplot(3, 2, star)

            # Get the log probabilities for this star rating
            star_probs = classifier.feature_log_prob_[star - 1]

            # Create DataFrame for visualization
            star_features = (
                pd.DataFrame({"Feature": feature_names, "Log Probability": star_probs})
                .sort_values("Log Probability", ascending=False)
                .head(n_features)
            )

            # Plot the most important features
            sns.barplot(
                x="Log Probability", y="Feature", data=star_features, palette="viridis"
            )
            plt.title(f"Top Features for {star} Stars")

        plt.tight_layout()
        plt.savefig("feature_importance.png")

    def predict_stars(self, reviews):
        """Predict star ratings for new reviews"""
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
            predicted_stars = predictions[i]
            confidence = probabilities[i][
                predicted_stars - 1
            ]  # -1 because stars are 1-5 but array is 0-4
            results.append(
                {
                    "review": review,
                    "predicted_stars": predicted_stars,
                    "confidence": confidence,
                }
            )

        return results

    def init(self):
        # Check if model already exists
        if os.path.exists(MODEL_PATH):
            print(f"Loading existing model from {MODEL_PATH}")
            with open(MODEL_PATH, "rb") as f:
                self.model = pickle.load(f)
        else:
            # Load and prepare data
            df = self.load_data()
            prepared_df = self.prepare_data(df)

            # Train and evaluate the model
            self.train_model(prepared_df)

    def predict_stars(self, reviews):
        """Predict star ratings for new reviews"""
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
            predicted_stars = predictions[i]
            confidence = probabilities[i][predicted_stars - 1]
            results.append(
                {
                    "review": review,
                    "predicted_stars": predicted_stars,
                    "confidence": confidence,
                }
            )

        return results


# Example usage
if __name__ == "__main__":
    # Initialize the analyzer
    analyzer = SentimentAnalyzerNB(DATA_SET)

    # Load data and train model
    analyzer.init()

    # # Analyze feature importance for stats
    # analyzer.analyze_feature_importance()

    # Example with manual predictions
    test_reviews = [
        "The food was amazing and the service was excellent!",
        "Worst experience ever. The staff was rude and the food was cold.",
        "The ambiance was nice but the food was just okay.",
        "Never had a greater dinner in my life. Go !!!",
    ]

    results = analyzer.predict_stars(test_reviews)

    print("\nExample Predictions:")
    for result in results:
        print(f"Review: {result['review']}")
        print(
            f"-> Predicted Stars: {result['predicted_stars']} (Confidence: {result['confidence']:.4f})"
        )

"""
DATASET SIZE INCREASED AND PREDICTING STARS - 1M REVIEW DATASET
Data Loading Summary:
Total reviews loaded: 1000000
Columns available: ['review_id', 'user_id', 'business_id', 'stars', 'useful', 'funny', 'cool', 'text', 'date']
Sample of star ratings distribution:
stars
1    138625
2     77912
3    102954
4    221897
5    458612
Name: count, dtype: int64

Data Preparation Summary:
Original text samples: 1000000
Warning: 33 reviews became empty after cleaning
Final dataset size: 999967 reviews

Star Rating Distribution:
1 stars: 138620 reviews (13.9%)
2 stars: 77907 reviews (7.8%)
3 stars: 102952 reviews (10.3%)
4 stars: 221886 reviews (22.2%)
5 stars: 458602 reviews (45.9%)

Training Data Summary:
Training samples: 799973
Testing samples: 199994

Training set star distribution:
stars
1    110896
2     62326
3     82361
4    177509
5    366881
Name: count, dtype: int64

Training model...
Training completed in 73.78 seconds.
Vocabulary size: 5000

Model Evaluation:
Accuracy: 0.6332

Classification Report:
              precision    recall  f1-score   support

           1       0.66      0.77      0.71     27724
           2       0.43      0.18      0.25     15581
           3       0.46      0.16      0.24     20591
           4       0.47      0.39      0.42     44377
           5       0.70      0.90      0.79     91721

    accuracy                           0.63    199994
   macro avg       0.54      0.48      0.48    199994
weighted avg       0.60      0.63      0.60    199994

Example Predictions:
Review: The food was amazing and the service was excellent!
-> Predicted Stars: 5 (Confidence: 0.9243)
Review: Worst experience ever. The staff was rude and the food was cold.
-> Predicted Stars: 1 (Confidence: 0.9736)
Review: The ambiance was nice but the food was just okay.
-> Predicted Stars: 3 (Confidence: 0.4891)
Review: Never had a greater dinner in my life. Go !!!
-> Predicted Stars: 5 (Confidence: 0.5560)
"""

"""
DATASET SIZE INCREASED AND PREDICTING STARS - 395K REVIEW PRE-PROCESSED DATASET
Data Loading Summary:
Total reviews loaded: 395630
Columns available: ['review_id', 'user_id', 'business_id', 'stars', 'useful', 'funny', 'cool', 'text', 'date']
Sample of star ratings distribution:
stars
1     70465
2     39836
3     51307
4     89099
5    144923
Name: count, dtype: int64

Data Preparation Summary:
Original text samples: 395630
Warning: 15 reviews became empty after cleaning
Final dataset size: 395615 reviews

Star Rating Distribution:
1 stars: 70460 reviews (17.8%)
2 stars: 39835 reviews (10.1%)
3 stars: 51303 reviews (13.0%)
4 stars: 89096 reviews (22.5%)
5 stars: 144921 reviews (36.6%)

Training Data Summary:
Training samples: 316492
Testing samples: 79123

Training set star distribution:
stars
1     56368
2     31868
3     41042
4     71277
5    115937
Name: count, dtype: int64

Training model...
Training completed in 24.14 seconds.
Vocabulary size: 5000

Model Evaluation:
Accuracy: 0.6151

Classification Report:
              precision    recall  f1-score   support

           1       0.68      0.82      0.74     14092
           2       0.46      0.20      0.28      7967
           3       0.47      0.25      0.32     10261
           4       0.47      0.42      0.45     17819
           5       0.68      0.88      0.77     28984

    accuracy                           0.62     79123
   macro avg       0.55      0.51      0.51     79123
weighted avg       0.58      0.62      0.58     79123


Example Predictions:
Review: The food was amazing and the service was excellent!
-> Predicted Stars: 5 (Confidence: 0.9493)
Review: Worst experience ever. The staff was rude and the food was cold.
-> Predicted Stars: 1 (Confidence: 0.9800)
Review: The ambiance was nice but the food was just okay.
-> Predicted Stars: 3 (Confidence: 0.4638)
Review: Never had a greater dinner in my life. Go !!!
-> Predicted Stars: 5 (Confidence: 0.4159)
"""
