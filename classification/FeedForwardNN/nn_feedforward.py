import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import json
import re
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from torch.utils.data import Dataset, DataLoader
from collections import Counter
import nltk
from nltk.corpus import stopwords
import matplotlib.pyplot as plt

# Download NLTK resources if needed
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# Define a PyTorch Dataset
class ReviewDataset(Dataset):
    def __init__(self, texts, ratings, vocab, max_length=200):
        self.texts = texts
        self.ratings = ratings
        self.vocab = vocab
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        rating = self.ratings[idx]
        
        # Convert text to sequence of indices
        tokens = self.tokenize(text)
        seq = [self.vocab.get(token, self.vocab['<UNK>']) for token in tokens]
        
        # Pad or truncate to max_length
        if len(seq) < self.max_length:
            seq = seq + [self.vocab['<PAD>']] * (self.max_length - len(seq))
        else:
            seq = seq[:self.max_length]
            
        return torch.tensor(seq, dtype=torch.long), torch.tensor(rating - 1, dtype=torch.long)  # 0-indexed classes
    
    def tokenize(self, text):
        # Simple tokenization
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
        return text.split()

# Neural Network Model
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

# Function to load and preprocess the data
def load_data(file_path):
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    
    df = pd.DataFrame(data)
    return df

# Preprocess and build vocabulary
def preprocess_data(df):
    # Keep only necessary columns
    df = df[['text', 'stars']]
    
    # Basic text preprocessing
    df['text'] = df['text'].str.lower()
    
    # Build vocabulary
    stop_words = set(stopwords.words('english'))
    all_words = []
    
    for text in df['text']:
        text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
        words = text.split()
        all_words.extend([w for w in words if w not in stop_words])
    
    # Get most common words
    word_counts = Counter(all_words)
    vocab_size = 10000  # Use top 10000 words
    most_common = word_counts.most_common(vocab_size - 2)  # -2 for <PAD> and <UNK>
    
    # Create vocabulary dictionary
    vocab = {'<PAD>': 0, '<UNK>': 1}
    for word, _ in most_common:
        vocab[word] = len(vocab)
    
    return df, vocab

# Training function
def train_model(train_loader, model, optimizer, criterion, device):
    model.train()
    total_loss = 0
    
    for texts, labels in train_loader:
        texts, labels = texts.to(device), labels.to(device)
        
        optimizer.zero_grad()
        predictions = model(texts)
        loss = criterion(predictions, labels)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(train_loader)

# Evaluation function
def evaluate_model(data_loader, model, criterion, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for texts, labels in data_loader:
            texts, labels = texts.to(device), labels.to(device)
            
            predictions = model(texts)
            loss = criterion(predictions, labels)
            
            total_loss += loss.item()
            
            preds = torch.argmax(predictions, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
    
    accuracy = accuracy_score(all_labels, all_preds)
    report = classification_report(all_labels, all_preds)
    
    return total_loss / len(data_loader), accuracy, report

# Main function
def main():
    # Set up your model parameters
    EMBEDDING_DIM = 100
    HIDDEN_DIM = 256
    OUTPUT_DIM = 5
    N_LAYERS = 2
    DROPOUT = 0.5
    BATCH_SIZE = 64
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 10

    # Check if CUDA is available
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load and preprocess data
    print("Loading and preprocessing data...")
    # Replace with your actual data path
    file_path = "data_set/reviews.pkl"  
    
    # # For demonstration, let's create a small sample dataset
    # sample_data = [
    #     {"review_id": "YziM_B_tSwtoMnM6Peew7w", "stars": 4.0,
    #      "text": "My girlfriend and I stopped by in Boise for a night and decided to give the fork a try... this shit is delicious"},
    #     {"review_id": "sample2", "stars": 5.0, 
    #      "text": "Absolutely amazing service and food! The chef is incredible and I can't wait to come back."},
    #     {"review_id": "sample3", "stars": 2.0, 
    #      "text": "Food was okay but service was extremely slow. Waited over an hour for our entrees."},
    #     {"review_id": "sample4", "stars": 1.0, 
    #      "text": "Terrible experience. The food was cold and they charged us for items we didn't order."},
    #     {"review_id": "sample5", "stars": 3.0, 
    #      "text": "Nothing special but nothing terrible either. Pretty average place with average food."}
    # ]
    
    df = pd.read_pickle(file_path)
    df, vocab = preprocess_data(df)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        df['text'].values, df['stars'].values, test_size=0.2, random_state=42
    )
    
    # Create datasets and dataloaders
    train_dataset = ReviewDataset(X_train, y_train, vocab)
    test_dataset = ReviewDataset(X_test, y_test, vocab)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
    
    # Initialize model
    model = ReviewClassifier(
        len(vocab), EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM, N_LAYERS, DROPOUT
    ).to(device)
    
    # Define optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()
    
    # Training loop
    print("Starting training...")
    train_losses = []
    test_losses = []
    test_accuracies = []
    
    for epoch in range(NUM_EPOCHS):
        # Train
        train_loss = train_model(train_loader, model, optimizer, criterion, device)
        train_losses.append(train_loss)
        
        # Evaluate
        test_loss, test_accuracy, test_report = evaluate_model(test_loader, model, criterion, device)
        test_losses.append(test_loss)
        test_accuracies.append(test_accuracy)
        
        print(f"Epoch: {epoch+1}/{NUM_EPOCHS}")
        print(f"Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")
    
    print("\nTraining complete!")
    print("\nClassification Report:")
    print(test_report)
    
    # Plot training progress
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
    
    # Example prediction function
    def predict_rating(review_text, model, vocab, max_length=200):
        model.eval()
        
        # Preprocess and tokenize
        text = review_text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        tokens = text.split()
        
        # Convert to indices
        seq = [vocab.get(token, vocab['<UNK>']) for token in tokens]
        
        # Pad or truncate
        if len(seq) < max_length:
            seq = seq + [vocab['<PAD>']] * (max_length - len(seq))
        else:
            seq = seq[:max_length]
        
        # Convert to tensor
        seq_tensor = torch.tensor([seq], dtype=torch.long).to(device)
        
        # Get prediction
        with torch.no_grad():
            output = model(seq_tensor)
            predicted_class = torch.argmax(output, dim=1).item()
        
        return predicted_class + 1  # Convert back to 1-5 scale
    
    # Test with a new review
    test_review = "The food was pretty good but the service was extremely slow. I might come back again."
    predicted_rating = predict_rating(test_review, model, vocab)
    print(f"\nTest Review: {test_review}")
    print(f"Predicted Rating: {predicted_rating} stars")
    
    # Save the model
    torch.save({
        'model_state_dict': model.state_dict(),
        'vocab': vocab,
        'config': {
            'embedding_dim': EMBEDDING_DIM,
            'hidden_dim': HIDDEN_DIM,
            'output_dim': OUTPUT_DIM,
            'n_layers': N_LAYERS,
            'dropout': DROPOUT
        }
    }, 'review_classifier_model.pth')
    
    print("\nModel saved to 'review_classifier_model.pth'")

# if __name__ == "__main__":
#     main()

# Example of loading and using the saved model
def load_and_use_model(model_path, review_text):
    # Load the saved model
    checkpoint = torch.load(model_path)
    vocab = checkpoint['vocab']
    config = checkpoint['config']
    
    # Initialize the model with the same parameters
    model = ReviewClassifier(
        len(vocab), 
        config['embedding_dim'], 
        config['hidden_dim'], 
        config['output_dim'], 
        config['n_layers'], 
        config['dropout']
    )
    
    # Load the saved parameters
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Set to evaluation mode
    model.eval()
    
    # Preprocess the review
    text = review_text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    tokens = text.split()
    
    # Convert to indices
    max_length = 200
    seq = [vocab.get(token, vocab['<UNK>']) for token in tokens]
    
    # Pad or truncate
    if len(seq) < max_length:
        seq = seq + [vocab['<PAD>']] * (max_length - len(seq))
    else:
        seq = seq[:max_length]
    
    # Convert to tensor
    seq_tensor = torch.tensor([seq], dtype=torch.long)
    
    # Get prediction
    with torch.no_grad():
        output = model(seq_tensor)
        predicted_class = torch.argmax(output, dim=1).item()
    
    return predicted_class + 1  # Convert back to 1-5 scale

# Example usage
predicted_rating = load_and_use_model('review_classifier_model.pth', 
                                     "98")
print(f"Predicted Rating: {predicted_rating} stars")