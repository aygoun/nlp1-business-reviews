import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Load the dataset
df = pd.read_json('data_set/yelp_subset_review.json', lines=True)

df['label'] = df['stars'].apply(lambda x: 1 if x > 3.5 else 0)  # 1 for positive, 0 for negative

# Split the data into train and test sets
X = df['text']  # Reviews
y = df['label']   # Labels (positive or negative)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert text to numeric data using TF-IDF
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Logistic Regression Model
model = LogisticRegression()
model.fit(X_train_tfidf, y_train)

y_pred = model.predict(X_test_tfidf)

# Predict categories for new reviews
def classify_category(review):
    X_new = vectorizer.transform([review])
    return model.predict(X_new)[0]

# New reviews
new_review = "Never had a greater dinner in my life. Go !!!"
predicted_label = classify_category(new_review)
print(f"Predicted Label for the review: {predicted_label} (1 = Positive, 0 = Negative)")

new_review = "The food was amazing and the service was excellent!"
predicted_label = classify_category(new_review)
print(f"Predicted Label for the review: {predicted_label} (1 = Positive, 0 = Negative)")

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

print("Classification Report:")
print(classification_report(y_test, y_pred))
