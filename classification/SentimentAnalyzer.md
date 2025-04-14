# Sentiment Analyzer for Business Reviews

## Features

- Text preprocessing with NLTK
- TF-IDF vectorization
- Naive Bayes classification
- Feature importance analysis
- Visualization of results
- Support for both JSON and JSON Lines input formats

## How It Works

### 1. Text Preprocessing

The `Tokenizer` class in `tokenizer.py` handles text preprocessing:

- Converts text to lowercase
- Removes special characters and digits
- Tokenizes the text
- Removes stopwords
- Applies lemmatization to reduce words to their base form

### 2. Data Loading and Preparation

The `SentimentAnalyzerNB` class:

- Loads review data from JSON/JSON Lines format
- Preprocesses the text using the Tokenizer
- Converts star ratings to binary sentiment labels (positive/negative)
- Splits data into training and test sets

### 3. Model Training

The system uses a pipeline with two main components:

1. TF-IDF Vectorizer:

   - Converts text to numerical features
   - Uses 1-2 word combinations (ngrams)
   - Limits to top 5000 features

2. Multinomial Naive Bayes Classifier:
   - Trains on the vectorized text
   - Uses Laplace smoothing (alpha=0.1)
   - Classifies reviews as positive or negative

### 4. Feature Analysis

The system provides insights into feature importance:

- Analyzes top contributing words for both positive and negative sentiments
- Visualizes feature importance using bar plots
- Saves visualization as "feature_importance.png"

### 5. Prediction

The system can predict sentiment for new reviews:

- Preprocesses input text
- Returns sentiment classification (Positive/Negative)
- Provides confidence scores for predictions

## Usage

### Basic Usage

```python
from classification.SentimentAnalyzerNB import SentimentAnalyzerNB

# Initialize analyzer
analyzer = SentimentAnalyzerNB("path/to/reviews.json", threshold=3.5)

# Load data and train model
analyzer.init()

# Analyze feature importance
analyzer.analyze_feature_importance()

# Make predictions
reviews = [
    "The food was amazing and the service was excellent!",
    "Worst experience ever. The staff was rude and the food was cold."
]
results = analyzer.predict_sentiment(reviews)
```

### Configuration Options

- `threshold`: Star rating threshold for positive/negative classification (default: 3.0)
- `n_features`: Number of top features to display in analysis (default: 20)

## Output

The system generates:

1. Model performance metrics:

   - Accuracy
   - Classification report (precision, recall, F1-score)
   - Confusion matrix visualization

2. Feature importance visualization:

   - Bar plot showing top contributing words
   - Separate colors for positive and negative features
   - Saved as "feature_importance.png"

3. Prediction results:
   - Sentiment classification
   - Confidence scores
   - Original review text

## Example Output

```
Example Predictions:
- Review: The food was amazing and the service was excellent!
-> Sentiment: Positive (Confidence: 0.9234)
- Review: Worst experience ever. The staff was rude and the food was cold.
-> Sentiment: Negative (Confidence: 0.8765)
```
