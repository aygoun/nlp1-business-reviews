import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional, SpatialDropout1D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


class YelpReviewClassifier:
    def __init__(self, max_features=5000, maxlen=200, embedding_dims=128):
        """
        Initialize the Yelp Review Classifier.
        
        Args:
            max_features (int): Size of the vocabulary
            maxlen (int): Maximum length of the sequences
            embedding_dims (int): Dimension of the embedding layer
        """
        # Download NLTK resources
        nltk.download('stopwords')
        nltk.download('wordnet')
        
        # Model parameters
        self.max_features = max_features
        self.maxlen = maxlen
        self.embedding_dims = embedding_dims
        
        # Initialize components
        self.tokenizer = None
        self.model = None
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
    
    def clean_text(self, text):
        """
        Clean the text by removing HTML tags, URLs, special characters, and numbers.
        
        Args:
            text (str): Input text
            
        Returns:
            str: Cleaned text
        """
        # Convert to lowercase
        text = text.lower()
        # Remove HTML tags
        text = re.sub(r'<.*?>', '', text)
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)
        # Remove special characters and numbers
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\d+', '', text)
        # Remove extra spaces
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    def preprocess_text(self, text, remove_stopwords=True):
        """
        Preprocess the text by cleaning and optionally removing stopwords and lemmatizing.
        
        Args:
            text (str): Input text
            remove_stopwords (bool): Whether to remove stopwords and lemmatize
            
        Returns:
            str: Preprocessed text
        """
        # Clean the text
        text = self.clean_text(text)
        
        if remove_stopwords:
            words = text.split()
            words = [self.lemmatizer.lemmatize(word) for word in words if word not in self.stop_words]
            text = ' '.join(words)
        
        return text
    
    def load_and_prepare_data(self, json_file_path):
        """
        Load data from a JSON file and prepare it for training.
        
        Args:
            json_file_path (str): Path to the JSON file
            
        Returns:
            tuple: Processed data and labels
        """
        # Load data
        df = pd.read_json(json_file_path, lines=True)
        
        print(f"Total number of reviews: {len(df)}")
        print(f"Star rating distribution:\n{df['stars'].value_counts(normalize=True).sort_index() * 100}")
        
        # Preprocess text
        df['processed_review'] = df['text'].apply(self.preprocess_text)
        
        # Analyze review length
        df['review_length'] = df['processed_review'].apply(lambda x: len(x.split()))
        
        # Prepare data for model
        X = df['processed_review'].values
        y = df['stars'].values
        
        return X, y
    
    def prepare_data_splits(self, X, y, test_size=0.3, val_size=0.15):
        """
        Split the data into training, validation, and test sets.
        
        Args:
            X (numpy.ndarray): Features
            y (numpy.ndarray): Labels
            test_size (float): Proportion of data for the test set
            val_size (float): Proportion of data for the validation set
            
        Returns:
            tuple: Training, validation, and test sets
        """
        # Split into train and temp sets
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # Split temp into validation and test sets
        test_ratio = val_size / test_size
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
        )
        
        # Create tokenizer
        self.tokenizer = Tokenizer(num_words=self.max_features)
        self.tokenizer.fit_on_texts(X_train)
        
        # Convert texts to sequences
        X_train_seq = self.tokenizer.texts_to_sequences(X_train)
        X_val_seq = self.tokenizer.texts_to_sequences(X_val)
        X_test_seq = self.tokenizer.texts_to_sequences(X_test)
        
        # Pad sequences
        X_train_pad = pad_sequences(X_train_seq, maxlen=self.maxlen)
        X_val_pad = pad_sequences(X_val_seq, maxlen=self.maxlen)
        X_test_pad = pad_sequences(X_test_seq, maxlen=self.maxlen)
        
        return (X_train_pad, y_train), (X_val_pad, y_val), (X_test_pad, y_test)
    
    def build_model(self):
        """
        Build the RNN model architecture.
        """
        vocab_size = min(self.max_features, len(self.tokenizer.word_index) + 1)
        
        self.model = Sequential()
        self.model.add(Embedding(vocab_size, self.embedding_dims))
        self.model.add(SpatialDropout1D(0.2))
        self.model.add(Bidirectional(LSTM(64, return_sequences=True)))
        self.model.add(Bidirectional(LSTM(32)))
        self.model.add(Dense(32, activation='relu'))
        self.model.add(Dropout(0.3))
        self.model.add(Dense(6, activation='softmax'))
        
        self.model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.model.summary()
        
        return self.model
    
    def train(self, train_data, val_data, epochs=10, batch_size=64, checkpoint_path='best_model.keras'):
        """
        Train the model.
        
        Args:
            train_data (tuple): Training data and labels
            val_data (tuple): Validation data and labels
            epochs (int): Number of epochs
            batch_size (int): Batch size
            checkpoint_path (str): Path to save the best model
            
        Returns:
            History object
        """
        X_train, y_train = train_data
        X_val, y_val = val_data
        
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=3,
            restore_best_weights=True
        )
        
        model_checkpoint = ModelCheckpoint(
            checkpoint_path,
            monitor='val_accuracy',
            save_best_only=True,
            mode='max'
        )
        
        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_val, y_val),
            callbacks=[early_stopping, model_checkpoint],
            verbose=1
        )
        
        return history
    
    def evaluate(self, test_data):
        """
        Evaluate the model on the test set.
        
        Args:
            test_data (tuple): Test data and labels
            
        Returns:
            dict: Evaluation metrics
        """
        X_test, y_test = test_data
        
        y_pred_prob = self.model.predict(X_test)
        y_pred = np.argmax(y_pred_prob, axis=1)
        
        # Metrics
        accuracy = accuracy_score(y_test, y_pred)
        print(f"\nTest accuracy: {accuracy:.4f}")
        
        # Confusion matrix
        plt.figure(figsize=(10, 8))
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=range(6), yticklabels=range(6))
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        plt.show()
        
        # Classification report
        print("\nClassification Report:")
        target_names = ['0 stars', '1 star', '2 stars', '3 stars', '4 stars']
        report = classification_report(y_test, y_pred, target_names=target_names)
        print(report)
        
        return {
            'accuracy': accuracy,
            'confusion_matrix': cm,
            'classification_report': report
        }
    
    def predict_stars(self, text):
        """
        Predict the rating for a new review.
        
        Args:
            text (str): Review text
            
        Returns:
            tuple: Predicted rating, confidence, and probabilities
        """
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model or tokenizer not initialized. Train the model first.")
        
        processed_text = self.preprocess_text(text)
        
        sequence = self.tokenizer.texts_to_sequences([processed_text])
        padded = pad_sequences(sequence, maxlen=self.maxlen)
        
        prediction = self.model.predict(padded)[0]
        rating = np.argmax(prediction)
        confidence = prediction[rating]
        
        return rating#, confidence, prediction

    def save_model_and_tokenizer(self, path='./models/yelp_classifier_bundle.pkl'):
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model or tokenizer not initialized. Train the model first.")

        bundle = {
            'model': self.model,
            'tokenizer': self.tokenizer
        }
        with open(path, 'wb') as f:
            pickle.dump(bundle, f)
        print(f"Model and tokenizer saved together to {path}")

    def load_model_and_tokenizer(self, path='./models/yelp_classifier_bundle.pkl'):
        with open(path, 'rb') as f:
            bundle = pickle.load(f)
            self.model = bundle['model']
            self.tokenizer = bundle['tokenizer']
        print(f"Model and tokenizer loaded from {path}")

# Example usage
if __name__ == "__main__":
    # Initialize the classifier
    classifier = YelpReviewClassifier()
    
    # Load and prepare data
    X, y = classifier.load_and_prepare_data('/kaggle/input/review/yelp_subset_review.json')
    
    # Split data
    train_data, val_data, test_data = classifier.prepare_data_splits(X, y)
    
    # Build model
    classifier.build_model()
    
    # Train model
    history = classifier.train(train_data, val_data)
    
    # Evaluate model
    metrics = classifier.evaluate(test_data)
    
    # Test on sample reviews
    test_reviews = [
        "Came here for a business dinner seminar. The staff was awesome and the food and cocktails were really good too.",
        "Never had a greater dinner in my life. Go !!!",
        "If you want to pay for everything a la carte this is the place for you. Food wasn't terrible not impressive.",
        "I am a long term frequent customer of this establishment. I just went in to order take out (3 apps) and was told they're too busy to do it."
    ]
    
    print("\nPredictions on examples:")
    for review in test_reviews:
        rating, confidence, probs = classifier.predict(review)
        print(f"Review: {review[:100]}...")
        print(f"Predicted rating: {rating} stars (confidence: {confidence:.2f})")
        print(f"Probabilities by class: {probs}")
        print("-" * 50)
    
    # Save model only (without tokenizer)
    classifier.save_model_and_tokenizer()    

    '''
    classifier = YelpReviewClassifier()
    classifier.load_model_and_tokenizer()

    review_text = "The food was absolutely amazing and the service was great!"
    rating, confidence, probs = classifier.predict(review_text)
    print(f"Predicted Rating: {rating}, Confidence: {confidence:.2f}")
    '''