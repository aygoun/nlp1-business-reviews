import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pickle
import sys
import os

# Add the project root directory to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

from utils.classifier import aspect_keywords

extracted_data_set = "./data_set/yelp_subset_review.json"


# Function to label aspects based on keywords
def label_aspects(text):
    text = text.lower()
    aspects = []
    for aspect, keywords in aspect_keywords.items():
        if any(keyword in text for keyword in keywords):
            aspects.append(aspect)
    return aspects


# Load the smaller dataset
data = pd.read_json(extracted_data_set, lines=True)

# Create aspect labels for each review
data["aspects"] = data["text"].apply(label_aspects)

# Create binary columns for each aspect
for aspect in aspect_keywords.keys():
    data[aspect] = data["aspects"].apply(lambda x: 1 if aspect in x else 0)

# Convert the text to a vector
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(data["text"])

# Create a dictionary to store models for each aspect
aspect_models = {}

# Train separate models for each aspect
for aspect in aspect_keywords.keys():
    print(f"\nTraining model for aspect: {aspect}")

    # Split the data for this aspect
    y = data[aspect]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train the model
    model = MultinomialNB()
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Evaluate the model
    print(f"\nResults for {aspect}:")
    print(classification_report(y_test, y_pred))

    # Store the model
    aspect_models[aspect] = model

# Save the models and vectorizer
with open("aspect_models.pkl", "wb") as f:
    pickle.dump(aspect_models, f)
with open("aspect_vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)


# Function to analyze a new review
def analyze_review(text):
    # Transform the text
    X_new = vectorizer.transform([text])

    # Get predictions for each aspect
    results = {}
    for aspect, model in aspect_models.items():
        prob = model.predict_proba(X_new)[0][1]
        results[aspect] = {
            "mentioned": bool(model.predict(X_new)[0]),
            "probability": prob,
        }
    return results


# Example usage
print("\nExample review analysis:")
example_review = "Good restaurant! I recommend it to everyone!"
results = analyze_review(example_review)
for aspect, result in results.items():
    print(
        f"{aspect}: {'Mentioned' if result['mentioned'] else 'Not mentioned'} (Probability: {result['probability']:.2f})"
    )
