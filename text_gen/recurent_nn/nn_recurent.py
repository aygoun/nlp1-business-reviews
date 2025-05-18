import os
import torch
import torch.nn as nn
import torch.optim as optim
import nltk
import numpy as np
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer
from collections import Counter
from torch.utils.data import Dataset, DataLoader
from tokenizer.tokenizer import Tokenizer
from sklearn.model_selection import train_test_split

# Download NLTK data
nltk.download("sentence_polarity", quiet=True)
nltk.download("punkt", quiet=True)
from nltk.corpus import sentence_polarity

# Constants
MODEL_PATH = 'rnn_text_gen.pt'
EMBEDDING_DIM = 128
HIDDEN_DIM = 256
NUM_LAYERS = 2
DROPOUT = 0.2
BATCH_SIZE = 64
EPOCHS = 10
MAX_LENGTH = 50
EARLY_STOPPING_PATIENCE = 3

# Device setup
device = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Custom dataset
class TextDataset(Dataset):
    def __init__(self, sequences, targets):
        self.sequences = sequences
        self.targets = targets
        
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx]

# RNN model
class RNNModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, dropout):
        super(RNNModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(
            embedding_dim, 
            hidden_dim, 
            num_layers=num_layers, 
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        
    def forward(self, x):
        # x shape: [batch_size, seq_length]
        embedded = self.embedding(x)
        # embedded shape: [batch_size, seq_length, embedding_dim]
        
        output, _ = self.lstm(embedded)
        # output shape: [batch_size, seq_length, hidden_dim]
        
        # Get the output for the last time step
        output = output[:, -1, :]
        # output shape: [batch_size, hidden_dim]
        
        output = self.dropout(output)
        logits = self.fc(output)
        # logits shape: [batch_size, vocab_size]
        
        return logits

# Prepare data with train/val split
def prepare_data():
    # Get sentences from NLTK corpus
    sentences = [" ".join(sent) for sent in sentence_polarity.sents()]
    print(f"Total sentences: {len(sentences)}")
    
    # Initialize custom tokenizer
    tokenizer = Tokenizer()
    
    # Build vocabulary
    word_counts = Counter()
    for sentence in sentences:
        tokens = tokenizer.get_tokens(sentence)
        word_counts.update(tokens)
    
    # Create vocabulary with special tokens
    vocab = {'<PAD>': 0, '<UNK>': 1, '<BOS>': 2, '<EOS>': 3}
    for word, _ in word_counts.most_common():
        if word not in vocab:
            vocab[word] = len(vocab)
    
    # Create token to index mapping
    token_to_index = vocab
    index_to_token = {idx: token for token, idx in token_to_index.items()}
    vocab_size = len(token_to_index)
    print(f"Vocabulary size: {vocab_size}")
    
    # Create sequences for training
    all_sequences = []
    all_targets = []
    
    for sentence in sentences:
        tokens = tokenizer.get_tokens(sentence)
        if not tokens:  # Skip empty token lists
            continue
            
        token_indices = [token_to_index.get(token, token_to_index['<UNK>']) for token in tokens]
        
        for i in range(1, len(token_indices)):
            input_seq = token_indices[:i]
            target = token_indices[i]
            
            # Pad sequences to uniform length
            if len(input_seq) < MAX_LENGTH:
                pad_length = MAX_LENGTH - len(input_seq)
                input_seq = [token_to_index['<PAD>']] * pad_length + input_seq
            else:
                input_seq = input_seq[-MAX_LENGTH:]
            
            all_sequences.append(input_seq)
            all_targets.append(target)
    
    # Convert to tensors
    all_sequences = torch.tensor(all_sequences, dtype=torch.long)
    all_targets = torch.tensor(all_targets, dtype=torch.long)
    
    # Split into train and validation sets (80/20)
    train_sequences, val_sequences, train_targets, val_targets = train_test_split(
        all_sequences, all_targets, test_size=0.2, random_state=42
    )
    
    print(f"Training examples: {len(train_sequences)}")
    print(f"Validation examples: {len(val_sequences)}")
    
    return (train_sequences, train_targets, 
            val_sequences, val_targets, 
            token_to_index, index_to_token, 
            vocab_size, tokenizer)

# Training function with validation
def train_model(model, train_data, val_data, token_to_index, index_to_token):
    train_sequences, train_targets = train_data
    val_sequences, val_targets = val_data
    
    # Create datasets and dataloaders
    train_dataset = TextDataset(train_sequences, train_targets)
    val_dataset = TextDataset(val_sequences, val_targets)
    
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop with early stopping
    best_val_loss = float('inf')
    patience_counter = 0
    
    model.train()
    for epoch in range(EPOCHS):
        # Training phase
        model.train()
        total_train_loss = 0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (batch_sequences, batch_targets) in enumerate(train_dataloader):
            # Move to device
            batch_sequences = batch_sequences.to(device)
            batch_targets = batch_targets.to(device)
            
            # Forward pass
            outputs = model(batch_sequences)
            loss = criterion(outputs, batch_targets)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            # Track statistics
            total_train_loss += loss.item()
            
            # Calculate accuracy
            _, predicted = torch.max(outputs, 1)
            train_total += batch_targets.size(0)
            train_correct += (predicted == batch_targets).sum().item()
            
            if batch_idx % 100 == 0:
                print(f"Epoch: {epoch+1}/{EPOCHS}, Batch: {batch_idx}/{len(train_dataloader)}, "
                      f"Loss: {loss.item():.4f}")
        
        # Calculate training metrics
        avg_train_loss = total_train_loss / len(train_dataloader)
        train_accuracy = train_correct / train_total * 100
        
        # Validation phase
        model.eval()
        total_val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch_sequences, batch_targets in val_dataloader:
                batch_sequences = batch_sequences.to(device)
                batch_targets = batch_targets.to(device)
                
                outputs = model(batch_sequences)
                loss = criterion(outputs, batch_targets)
                
                total_val_loss += loss.item()
                
                _, predicted = torch.max(outputs, 1)
                val_total += batch_targets.size(0)
                val_correct += (predicted == batch_targets).sum().item()
        
        # Calculate validation metrics
        avg_val_loss = total_val_loss / len(val_dataloader)
        val_accuracy = val_correct / val_total * 100
        
        # Print epoch statistics
        print(f"Epoch: {epoch+1}/{EPOCHS}")
        print(f"  Train - Loss: {avg_train_loss:.4f}, Accuracy: {train_accuracy:.2f}%")
        print(f"  Valid - Loss: {avg_val_loss:.4f}, Accuracy: {val_accuracy:.2f}%")
        
        # Early stopping check
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            
            # Save the best model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'token_to_index': token_to_index,
                'index_to_token': index_to_token
            }, MODEL_PATH)
            print(f"  Model saved to {MODEL_PATH}")
        else:
            patience_counter += 1
            print(f"  No improvement in validation loss. Patience: {patience_counter}/{EARLY_STOPPING_PATIENCE}")
            
            if patience_counter >= EARLY_STOPPING_PATIENCE:
                print("Early stopping triggered!")
                break
    
    # Load the best model
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    return model

# Improved text generation function with better error handling
def generate_text(model, seed_text, token_to_index, index_to_token, tokenizer, num_words=10, temperature=1.0):
    model.eval()
    
    # Tokenize seed text
    tokens = tokenizer.get_tokens(seed_text)
    if not tokens:  # Handle empty tokens
        return seed_text
        
    token_indices = [token_to_index.get(token, token_to_index['<UNK>']) for token in tokens]
    
    generated_text = seed_text
    
    with torch.no_grad():
        for _ in range(num_words):
            # Prepare input
            if len(token_indices) < MAX_LENGTH:
                padded_indices = [token_to_index['<PAD>']] * (MAX_LENGTH - len(token_indices)) + token_indices
            else:
                padded_indices = token_indices[-MAX_LENGTH:]
            
            # Convert to tensor
            input_tensor = torch.tensor([padded_indices], dtype=torch.long).to(device)
            
            # Get predictions
            output = model(input_tensor)
            
            # Apply temperature
            if temperature != 1.0:
                output = output / temperature
            
            # Convert to probabilities
            probs = torch.softmax(output, dim=1)
            
            # Sample from distribution
            if temperature == 0:
                # Greedy sampling
                predicted_idx = torch.argmax(probs, dim=1).item()
            else:
                # Temperature sampling
                predicted_idx = torch.multinomial(probs, 1).item()
            
            # Add to sequence
            token_indices.append(predicted_idx)
            
            # Get the corresponding word
            predicted_token = index_to_token.get(predicted_idx, '<UNK>')
            if predicted_token in ['<PAD>', '<UNK>', '<BOS>', '<EOS>']:
                continue
                
            # Add to generated text
            generated_text += " " + predicted_token
    
    return generated_text

# Improved evaluation function with better error handling
def evaluate_model(model, sentences, token_to_index, index_to_token, tokenizer, num_samples=100):
    bleu_scores = []
    rouge_scores = []
    
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    
    for sent in sentences[:num_samples]:
        words = sent.split()
        if len(words) < 4:
            continue  # Skip very short sentences
            
        # Use first 2 words as seed
        if len(words) < 3:  # Skip if not enough words
            continue
            
        seed = " ".join(words[:2])
        reference = " ".join(words[2:])
        
        try:
            generated = generate_text(
                model, seed, token_to_index, index_to_token, tokenizer, 
                num_words=len(words[2:])
            )
            
            # Skip if generation failed or is identical to seed
            if generated == seed:
                continue
                
            # Extract generated part (excluding seed)
            generated_words = generated.split()
            if len(generated_words) <= 2:
                continue
                
            generated_tail = " ".join(generated_words[2:])
            
            # Skip if generated tail is empty
            if not generated_tail:
                continue
                
            # Calculate BLEU score
            bleu = sentence_bleu(
                [reference.split()], 
                generated_tail.split(), 
                weights=(0.5, 0.5)
            )
            
            # Calculate ROUGE scores
            rouge = scorer.score(reference, generated_tail)
            
            bleu_scores.append(bleu)
            rouge_scores.append(rouge)
        except Exception as e:
            print(f"Error evaluating sentence: {e}")
            continue
    
    # Calculate metrics only if we have scores
    if bleu_scores:
        avg_bleu = np.mean(bleu_scores)
        avg_rouge1 = np.mean([score['rouge1'].fmeasure for score in rouge_scores])
        avg_rougeL = np.mean([score['rougeL'].fmeasure for score in rouge_scores])
        
        print(f"\nAverage BLEU: {avg_bleu:.4f}")
        print(f"Average ROUGE-1 F1: {avg_rouge1:.4f}")
        print(f"Average ROUGE-L F1: {avg_rougeL:.4f}")
    else:
        print("\nNo valid evaluations completed. Model may not be generating text properly.")

def main():
    # Prepare data with train/val split
    (train_sequences, train_targets,
     val_sequences, val_targets,
     token_to_index, index_to_token,
     vocab_size, tokenizer) = prepare_data()
    
    # Package data for convenience
    train_data = (train_sequences, train_targets)
    val_data = (val_sequences, val_targets)
    
    # Check if model exists
    if os.path.exists(MODEL_PATH):
        print(f"Loading model from {MODEL_PATH}...")
        checkpoint = torch.load(MODEL_PATH, map_location=device)
        model = RNNModel(vocab_size, EMBEDDING_DIM, HIDDEN_DIM, NUM_LAYERS, DROPOUT).to(device)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Safely load dictionaries
        token_to_index = checkpoint.get('token_to_index', token_to_index)
        index_to_token = checkpoint.get('index_to_token', index_to_token)
    else:
        print("Training new model...")
        model = RNNModel(vocab_size, EMBEDDING_DIM, HIDDEN_DIM, NUM_LAYERS, DROPOUT).to(device)
        model = train_model(model, train_data, val_data, token_to_index, index_to_token)
    
    # Generate text with different temperatures
    sentences = [" ".join(sent) for sent in sentence_polarity.sents()]
    
    print("\nGenerated text examples:")
    print(f"Temp=1.0: {generate_text(model, 'the restaurant', token_to_index, index_to_token, tokenizer, 10, 1.0)}")
    print(f"Temp=0.7: {generate_text(model, 'the restaurant', token_to_index, index_to_token, tokenizer, 10, 0.7)}")
    print(f"Temp=0.5: {generate_text(model, 'the restaurant', token_to_index, index_to_token, tokenizer, 10, 0.5)}")
    print(f"Temp=0.2: {generate_text(model, 'the restaurant', token_to_index, index_to_token, tokenizer, 10, 0.2)}")
    
    print("\nMore examples:")
    print(f"'I love': {generate_text(model, 'I love', token_to_index, index_to_token, tokenizer, 10, 0.7)}")
    print(f"'The movie': {generate_text(model, 'The movie', token_to_index, index_to_token, tokenizer, 10, 0.7)}")
    print(f"'This is': {generate_text(model, 'This is', token_to_index, index_to_token, tokenizer, 10, 0.7)}")
    
    # Evaluate the model
    print("\nEvaluating model...")
    evaluate_model(model, sentences, token_to_index, index_to_token, tokenizer)

if __name__ == "__main__":
    main()