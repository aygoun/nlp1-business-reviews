import torch
import torch.nn as nn
import numpy as np
import os
from tokenizer.tokenizer import Tokenizer

class ReviewClassifier(nn.Module):
    """Neural Network Model for Review Classification"""
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.fc1 = nn.Linear(embedding_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        
    def forward(self, text):
        # text shape: [batch size, sentence length]
        embedded = self.embedding(text)
        # embedded shape: [batch size, sentence length, embedding dim]
        
        # Take the mean of the word embeddings
        pooled = torch.mean(embedded, dim=1)
        # pooled shape: [batch size, embedding dim]
        
        x = self.relu(self.fc1(pooled))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        return self.fc3(x)

class ReviewStarPredictor:
    """
    A wrapper class for the review classification model that provides
    an easy-to-use interface for star rating prediction.
    """
    
    def __init__(self, model_path='review_classifier_model.pth'):
        """
        Initialize the ReviewStarPredictor.
        
        Args:
            model_path: Path to the saved model file
        """
        self.model_path = model_path
        self.max_length = 200
        
        # Initialize the tokenizer
        self.tokenizer = Tokenizer()
        
        # Check if model exists, if not raise an error
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at: {model_path}")
        
        # Load the model
        self._load_model()
        
        # Set device
        self.device = torch.device('mps' if torch.backends.mps.is_available() else 
                                  ('cuda' if torch.cuda.is_available() else 'cpu'))
        self.model = self.model.to(self.device)
        
        print(f"Review Star Predictor initialized using device: {self.device}")
        
    def _load_model(self):
        """Load the saved model and vocabulary"""
        # Load the checkpoint
        try:
            checkpoint = torch.load(self.model_path, map_location=torch.device('cpu'))
        except:
            # Try loading with weights_only parameter for older PyTorch versions
            checkpoint = torch.load(self.model_path, map_location=torch.device('cpu'), weights_only=True)
            
        self.vocab = checkpoint['vocab']
        config = checkpoint['config']
        
        # Initialize the model with the saved parameters
        self.model = ReviewClassifier(
            len(self.vocab), 
            config['embedding_dim'], 
            config['hidden_dim'], 
            config['output_dim'], 
            config['n_layers'], 
            config['dropout']
        )
        
        # Load the saved parameters
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Set model to evaluation mode
        self.model.eval()
        
    def _preprocess_text(self, text):
        """Preprocess the input text for prediction"""
        # Use custom tokenizer
        tokens = self.tokenizer.get_tokens(text)
        
        # Convert tokens to indices
        seq = [self.vocab.get(token, self.vocab['<UNK>']) for token in tokens]
        
        # Pad or truncate to max_length
        if len(seq) < self.max_length:
            seq = seq + [self.vocab['<PAD>']] * (self.max_length - len(seq))
        else:
            seq = seq[:self.max_length]
        
        # Convert to tensor
        return torch.tensor([seq], dtype=torch.long)
    
    def predict_stars(self, review_text):
        """
        Predict the star rating (1-5) for a given review text.
        
        Args:
            review_text: The text of the review to predict a rating for
            
        Returns:
            An integer from 1 to 5 representing the predicted star rating
        """
        # Ensure model is in evaluation mode
        self.model.eval()
        
        # Preprocess the text
        seq_tensor = self._preprocess_text(review_text).to(self.device)
        
        # Get prediction
        with torch.no_grad():
            output = self.model(seq_tensor)
            predicted_class = torch.argmax(output, dim=1).item()
        
        # Convert from 0-indexed to 1-5 star rating
        return predicted_class + 1
    
    def predict_stars_with_confidence(self, review_text):
        """
        Predict the star rating (1-5) with confidence scores for a given review text.
        
        Args:
            review_text: The text of the review to predict a rating for
            
        Returns:
            A tuple of (predicted_rating, confidence_scores) where confidence_scores
            is a dictionary mapping each star rating to its confidence score (0-1)
        """
        # Ensure model is in evaluation mode
        self.model.eval()
        
        # Preprocess the text
        seq_tensor = self._preprocess_text(review_text).to(self.device)
        
        # Get prediction
        with torch.no_grad():
            output = self.model(seq_tensor)
            probabilities = torch.softmax(output, dim=1)[0].cpu().numpy()
            predicted_class = np.argmax(probabilities)
        
        # Convert probabilities to confidence scores dictionary
        confidence_scores = {
            f"{i+1} stars": float(prob) 
            for i, prob in enumerate(probabilities)
        }
        
        # Convert from 0-indexed to 1-5 star rating
        return predicted_class + 1, confidence_scores
    
    def batch_predict(self, review_texts):
        """
        Predict star ratings for multiple reviews at once.
        
        Args:
            review_texts: List of review texts
            
        Returns:
            List of predicted star ratings (1-5)
        """
        # Ensure model is in evaluation mode
        self.model.eval()
        
        predictions = []
        for review in review_texts:
            # Process each review individually
            seq_tensor = self._preprocess_text(review).to(self.device)
            
            # Get prediction
            with torch.no_grad():
                output = self.model(seq_tensor)
                predicted_class = torch.argmax(output, dim=1).item()
            
            # Convert from 0-indexed to 1-5 star rating
            predictions.append(predicted_class + 1)
        
        return predictions


# Example usage
if __name__ == "__main__":
    # Initialize the predictor
    predictor = ReviewStarPredictor('review_classifier_model.pth')
    
    # Single prediction
    review = "The food was delicious but the service was slow."
    rating = predictor.predict_stars(review)
    print(f"Review: {review}")
    print(f"Predicted Rating: {rating} stars")
    
    # Prediction with confidence scores
    rating, confidence = predictor.predict_stars_with_confidence(review)
    print(f"Predicted Rating: {rating} stars")
    print(f"Confidence Scores:")
    for stars, score in confidence.items():
        print(f"  {stars}: {score:.4f}")
    
    # Batch prediction
    reviews = [
        "The food was amazing and the service was excellent!",
        "Terrible experience. Would not recommend.",
        "Average food at best. Nothing special."
    ]
    ratings = predictor.batch_predict(reviews)
    
    print("\nBatch Predictions:")
    for review, rating in zip(reviews, ratings):
        print(f"Review: {review}")
        print(f"Predicted Rating: {rating} stars")
        print()