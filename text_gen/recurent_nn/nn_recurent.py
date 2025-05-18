import os
import tensorflow as tf
import nltk

# We will use the NLTK sentence polarity corpus for training
nltk.download("sentence_polarity", quiet=True)
from nltk.corpus import sentence_polarity
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Libs for benchmarking
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer

MODEL_PATH = "./rnn_text_gen.keras"

# MacOS or CPU
device = "mps" if tf.config.list_physical_devices("mps") else "cpu"
print(f"Using device: {device}")

# Load sentences from NLTK polarity corpus
with tf.device(device):
    sentences = [" ".join(sent) for sent in sentence_polarity.sents()]

# Text preprocessing and sequence generation
tokenizer = Tokenizer()
tokenizer.fit_on_texts(sentences)
vocab_size = len(tokenizer.word_index) + 1
print(f"Vocabulary size: {vocab_size}")

# Create training sequences (predict next word)
X, y = [], []
for sent in sentences:
    token_list = tokenizer.texts_to_sequences([sent])[0]
    for i in range(1, len(token_list)):
        X.append(token_list[:i])
        y.append(token_list[i])

# Pad sequences to uniform length
max_length = max([len(seq) for seq in X])
X = pad_sequences(X, maxlen=max_length, padding="pre")
y = np.array(y)

print(f"Input shape: {X.shape}")
print(f"Output shape: {y.shape}")
print(f"Maximum sequence length: {max_length}")

if os.path.exists(MODEL_PATH):
    print(f"Loading model from {MODEL_PATH}...")
    model = tf.keras.models.load_model(MODEL_PATH)
else:
    print("Training new model...")
    with tf.device(device):
        # RNN model with LSTM
        input_layer = Input(shape=(max_length,))
        embedding = Embedding(input_dim=vocab_size, output_dim=128)(input_layer)
        
        # Replace Flatten layer with LSTM layer
        lstm = LSTM(256, return_sequences=False)(embedding)
        dropout = Dropout(0.2)(lstm)
        
        # Output layer
        output = Dense(vocab_size, activation="softmax")(dropout)
        
        # Create model
        model = Model(inputs=input_layer, outputs=output)
        model.compile(loss="sparse_categorical_crossentropy", 
                     optimizer="adam", 
                     metrics=["accuracy"])

    # Display model summary
    model.summary()
    
    # Train the model
    history = model.fit(
        X, y, 
        epochs=15, 
        batch_size=128, 
        validation_split=0.1,
        verbose=1
    )
    
    # Save the model
    model.save(MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")


def generate_text(seed_text, num_words=10, temperature=1.0):
    """
    Generate text using the trained RNN model.
    
    Args:
        seed_text: Initial text to start generation
        num_words: Number of words to generate
        temperature: Controls randomness (lower = more deterministic)
    
    Returns:
        Generated text including the seed text
    """
    for _ in range(num_words):
        # Tokenize the current text
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        
        # Pad the sequence
        token_list = pad_sequences([token_list], maxlen=max_length, padding="pre")
        
        # Predict the next word
        predictions = model.predict(token_list, verbose=0)[0]
        
        # Apply temperature scaling
        predictions = np.log(predictions) / temperature
        exp_preds = np.exp(predictions)
        predictions = exp_preds / np.sum(exp_preds)
        
        # Sample from the distribution
        if temperature == 0:  # Deterministic
            predicted = np.argmax(predictions)
        else:  # Sample with temperature
            predicted = np.random.choice(len(predictions), p=predictions)

        # Find the corresponding word
        output_word = ""
        for word, index in tokenizer.word_index.items():
            if index == predicted:
                output_word = word
                break
                
        # Add the word to the seed text
        seed_text += " " + output_word
        
    return seed_text


def evaluate_model(sentences, num_samples=100):
    """
    Evaluate the model using BLEU and ROUGE scores.
    """
    bleu_scores = []
    rouge_scores = []

    scorer = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)

    for sent in sentences[:num_samples]:
        words = sent.split()
        if len(words) < 4:
            continue  # skip very short sentences
        seed = " ".join(words[:2])
        reference = " ".join(words[2:])

        generated = generate_text(seed, num_words=len(words[2:]))
        generated_tail = " ".join(generated.split()[2:])  # remove seed from prediction

        bleu = sentence_bleu(
            [reference.split()], generated_tail.split(), weights=(0.5, 0.5)
        )
        rouge = scorer.score(reference, generated_tail)

        bleu_scores.append(bleu)
        rouge_scores.append(rouge)

    avg_bleu = np.mean(bleu_scores)
    avg_rouge1 = np.mean([score["rouge1"].fmeasure for score in rouge_scores])
    avg_rougeL = np.mean([score["rougeL"].fmeasure for score in rouge_scores])

    print(f"\nAverage BLEU: {avg_bleu:.4f}")
    print(f"Average ROUGE-1 F1: {avg_rouge1:.4f}")
    print(f"Average ROUGE-L F1: {avg_rougeL:.4f}")


# Example generation with different temperatures
print("\nGenerated text (temp=1.0):", generate_text("the movie", num_words=10, temperature=1.0))
print("Generated text (temp=0.7):", generate_text("the movie", num_words=10, temperature=0.7))
print("Generated text (temp=0.5):", generate_text("the movie", num_words=10, temperature=0.5))
print("Generated text (temp=0.2):", generate_text("the movie", num_words=10, temperature=0.2))

# Evaluate the model
print("\nEvaluating model...")
evaluate_model(sentences)

# Generate text with different starting phrases
print("\nMore examples:")
print("1:", generate_text("i love", num_words=15, temperature=0.7))
print("2:", generate_text("it was", num_words=15, temperature=0.7))
print("3:", generate_text("she said", num_words=15, temperature=0.7))