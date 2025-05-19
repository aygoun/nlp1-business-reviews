import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from torch.utils.data import Dataset, DataLoader
from collections import Counter
from tokenizer.tokenizer import Tokenizer

# Keep the original Dataset class
class ReviewDataset(Dataset):
    def __init__(self, texts, ratings, vocab, tokenizer, max_length=200):
        self.texts = texts
        self.ratings = ratings
        self.vocab = vocab
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        rating = self.ratings[idx]
        
        # Use custom tokenizer
        tokens = self.tokenizer.get_tokens(text)
        seq = [self.vocab.get(token, self.vocab['<UNK>']) for token in tokens]
        
        # Pad or truncate to max_length
        if len(seq) < self.max_length:
            seq = seq + [self.vocab['<PAD>']] * (self.max_length - len(seq))
        else:
            seq = seq[:self.max_length]
            
        return torch.tensor(seq, dtype=torch.long), torch.tensor(rating - 1, dtype=torch.long)  # 0-indexed classes

class ReviewClassifier(nn.Module):
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
    A wrapper class for the review classification model that exposes
    predict_stars() function and supports training with epochs.
    """
    
    def __init__(self, 
                model_path='review_classifier_model.pth',
                data_path='data_set/reviews.pkl',
                checkpoint_dir='./model_checkpoints',
                embedding_dim=100,
                hidden_dim=256,
                output_dim=5,
                n_layers=2,
                dropout=0.5,
                batch_size=64,
                learning_rate=0.001,
                epochs=10):
        """
        Initialize the ReviewStarPredictor.
        
        Args:
            model_path: Path to save/load the model
            data_path: Path to the dataset
            checkpoint_dir: Directory to save model checkpoints
            embedding_dim: Dimension of word embeddings
            hidden_dim: Number of hidden units
            output_dim: Number of output classes (5 for star ratings)
            n_layers: Number of layers
            dropout: Dropout rate
            batch_size: Batch size for training
            learning_rate: Learning rate for training
            epochs: Number of training epochs
        """
        # Create checkpoint directory if it doesn't exist
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Store parameters
        self.model_path = model_path
        self.data_path = data_path
        self.checkpoint_dir = checkpoint_dir
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.n_layers = n_layers
        self.dropout = dropout
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.max_length = 200
        
        # Initialize tokenizer
        self.tokenizer = Tokenizer()
        
        # Set device
        self.device = torch.device('mps' if torch.backends.mps.is_available() else 
                                 ('cuda' if torch.cuda.is_available() else 'cpu'))
        print(f"Using device: {self.device}")
        
        # Load model if it exists
        if os.path.exists(model_path):
            print(f"Loading model from {model_path}...")
            self._load_model()
        else:
            print(f"Model file not found at {model_path}. Use train() to train a new model.")
            self.model = None
            self.vocab = None
    
    def _load_model(self):
        """Load the saved model and vocabulary"""
        try:
            checkpoint = torch.load(self.model_path, map_location=self.device)
        except:
            # Try with weights_only for older PyTorch versions
            checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=True)
            
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
        ).to(self.device)
        
        # Load the saved parameters
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Set to evaluation mode
        self.model.eval()
    
    def _load_data(self):
        """Load and preprocess the dataset"""
        # Load data
        if self.data_path.endswith('.pkl'):
            df = pd.read_pickle(self.data_path)
        else:
            import json
            data = []
            with open(self.data_path, 'r') as f:
                for line in f:
                    data.append(json.loads(line))
            df = pd.DataFrame(data)
        
        # Keep only necessary columns
        df = df[['text', 'stars']].copy()
        
        return df
    
    def _build_vocab(self, texts):
        """Build vocabulary from texts"""
        # Use custom tokenizer to get tokens for all texts
        all_tokens = []
        for text in texts:
            tokens = self.tokenizer.get_tokens(text)
            all_tokens.extend(tokens)
        
        # Get most common tokens
        token_counts = Counter(all_tokens)
        vocab_size = 10000  # Use top 10000 tokens
        most_common = token_counts.most_common(vocab_size - 2)  # -2 for <PAD> and <UNK>
        
        # Create vocabulary dictionary
        vocab = {'<PAD>': 0, '<UNK>': 1}
        for token, _ in most_common:
            vocab[token] = len(vocab)
        
        return vocab
    
    def train(self, epochs=None):
        """
        Train the model on the dataset.
        
        Args:
            epochs: Number of epochs (overrides the default)
            
        Returns:
            Dictionary with training history (losses and accuracies)
        """
        # Override epochs if provided
        if epochs is not None:
            self.epochs = epochs
            
        # Load and preprocess data
        print("Loading and preprocessing data...")
        df = self._load_data()
        
        # Build vocabulary
        self.vocab = self._build_vocab(df['text'])
        print(f"Vocabulary size: {len(self.vocab)}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            df['text'].values, df['stars'].values, test_size=0.2, random_state=42
        )
        
        # Create datasets and dataloaders
        train_dataset = ReviewDataset(X_train, y_train, self.vocab, self.tokenizer, self.max_length)
        test_dataset = ReviewDataset(X_test, y_test, self.vocab, self.tokenizer, self.max_length)
        
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size)
        
        # Initialize model
        self.model = ReviewClassifier(
            len(self.vocab), 
            self.embedding_dim, 
            self.hidden_dim, 
            self.output_dim, 
            self.n_layers, 
            self.dropout
        ).to(self.device)
        
        # Define optimizer and loss function
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        criterion = nn.CrossEntropyLoss()
        
        # Training loop
        print(f"Starting training for {self.epochs} epochs...")
        train_losses = []
        test_losses = []
        test_accuracies = []
        
        for epoch in range(self.epochs):
            # Train
            self.model.train()
            total_loss = 0
            
            for texts, labels in train_loader:
                texts, labels = texts.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()
                predictions = self.model(texts)
                loss = criterion(predictions, labels)
                
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            train_loss = total_loss / len(train_loader)
            train_losses.append(train_loss)
            
            # Evaluate
            self.model.eval()
            total_loss = 0
            all_preds = []
            all_labels = []
            
            with torch.no_grad():
                for texts, labels in test_loader:
                    texts, labels = texts.to(self.device), labels.to(self.device)
                    
                    predictions = self.model(texts)
                    loss = criterion(predictions, labels)
                    
                    total_loss += loss.item()
                    
                    preds = torch.argmax(predictions, dim=1).cpu().numpy()
                    all_preds.extend(preds)
                    all_labels.extend(labels.cpu().numpy())
            
            test_loss = total_loss / len(test_loader)
            test_losses.append(test_loss)
            
            accuracy = accuracy_score(all_labels, all_preds)
            test_accuracies.append(accuracy)
            
            # Save checkpoint for this epoch
            checkpoint_path = os.path.join(self.checkpoint_dir, f"model_epoch_{epoch+1:02d}.pth")
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'vocab': self.vocab,
                'config': {
                    'embedding_dim': self.embedding_dim,
                    'hidden_dim': self.hidden_dim,
                    'output_dim': self.output_dim,
                    'n_layers': self.n_layers,
                    'dropout': self.dropout
                }
            }, checkpoint_path)
            
            print(f"Epoch {epoch+1}/{self.epochs} checkpoint saved to {checkpoint_path}")
            print(f"Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, Test Accuracy: {accuracy:.4f}")
        
        # Save final model
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'vocab': self.vocab,
            'config': {
                'embedding_dim': self.embedding_dim,
                'hidden_dim': self.hidden_dim,
                'output_dim': self.output_dim,
                'n_layers': self.n_layers,
                'dropout': self.dropout
            }
        }, self.model_path)
        
        print(f"\nTraining complete! Final model saved to {self.model_path}")
        
        # Final evaluation report
        report = classification_report(all_labels, all_preds)
        print("\nClassification Report:")
        print(report)
        
        # Try to plot if matplotlib is available
        try:
            import matplotlib.pyplot as plt
            
            plt.figure(figsize=(12, 5))
            
            plt.subplot(1, 2, 1)
            plt.plot(train_losses, label='Train Loss')
            plt.plot(test_losses, label='Test Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.title('Loss During Training')
            
            plt.subplot(1, 2, 2)
            plt.plot(test_accuracies)
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.title('Test Accuracy During Training')
            
            plt.tight_layout()
            plt.savefig('training_progress.png')
            plt.show()
        except ImportError:
            print("Matplotlib not available for plotting")
        
        return {
            'train_losses': train_losses,
            'test_losses': test_losses,
            'test_accuracies': test_accuracies,
            'final_report': report
        }
    
    def predict_stars(self, review_text):
        """
        Predict the star rating (1-5) for a given review text.
        
        Args:
            review_text: The text of the review
            
        Returns:
            An integer from 1 to 5 representing the predicted star rating
        """
        if self.model is None:
            raise ValueError("Model not loaded. Either load a model or train one first.")
            
        # Set model to evaluation mode
        self.model.eval()
        
        # Use custom tokenizer
        tokens = self.tokenizer.get_tokens(review_text)
        
        # Convert to indices
        seq = [self.vocab.get(token, self.vocab['<UNK>']) for token in tokens]
        
        # Pad or truncate
        if len(seq) < self.max_length:
            seq = seq + [self.vocab['<PAD>']] * (self.max_length - len(seq))
        else:
            seq = seq[:self.max_length]
        
        # Convert to tensor
        seq_tensor = torch.tensor([seq], dtype=torch.long).to(self.device)
        
        # Get prediction
        with torch.no_grad():
            output = self.model(seq_tensor)
            predicted_class = torch.argmax(output, dim=1).item()
        
        # Convert from 0-indexed to 1-5 star rating
        return predicted_class + 1
    
    def list_checkpoints(self):
        """
        List all available model checkpoints.
        
        Returns:
            List of checkpoint filenames
        """
        if os.path.exists(self.checkpoint_dir):
            return sorted([f for f in os.listdir(self.checkpoint_dir) if f.endswith('.pth')])
        return []
    
    def load_checkpoint(self, checkpoint_filename):
        """
        Load a specific checkpoint.
        
        Args:
            checkpoint_filename: Name of the checkpoint file
            
        Returns:
            True if successful, False otherwise
        """
        checkpoint_path = os.path.join(self.checkpoint_dir, checkpoint_filename)
        if os.path.exists(checkpoint_path):
            try:
                checkpoint = torch.load(checkpoint_path, map_location=self.device)
            except:
                checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
                
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
            ).to(self.device)
            
            # Load the saved parameters
            self.model.load_state_dict(checkpoint['model_state_dict'])
            
            # Set to evaluation mode
            self.model.eval()
            
            print(f"Loaded checkpoint: {checkpoint_filename}")
            return True
        
        print(f"Checkpoint not found: {checkpoint_filename}")
        return False


# Example usage
if __name__ == "__main__":
    # Create the predictor with 10 epochs (default)
    predictor = ReviewStarPredictor(epochs=10)
    
    # Train the model if it doesn't exist
    if not os.path.exists(predictor.model_path):
        predictor.train()
    
    # Example review
    review = "The food was delicious but the service was slow."
    
    # Predict star rating
    rating = predictor.predict_stars(review)
    print(f"Review: {review}")
    print(f"Predicted Rating: {rating} stars")

