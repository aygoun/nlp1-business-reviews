import os
import tensorflow as tf
import numpy as np
from tensorflow.keras.callbacks import ModelCheckpoint
import nltk

class TextGenerator:
    """
    A simple wrapper class for the RNN text generation model that provides an easy-to-use
    interface for text generation and saves model checkpoints at every epoch.
    """
    
    def __init__(self, 
                model_path="./rnn_text_gen.keras",
                checkpoint_dir="./checkpoints",
                epochs=15,
                batch_size=128,
                embedding_dim=128,
                lstm_units=256,
                dropout_rate=0.2):
        """
        Initialize the TextGenerator wrapper.
        
        Args:
            model_path: Path to save/load the model
            checkpoint_dir: Directory to save model checkpoints
            epochs: Number of training epochs
            batch_size: Training batch size
            embedding_dim: Dimension of word embeddings
            lstm_units: Number of LSTM units
            dropout_rate: Dropout rate for regularization
        """
        # Create checkpoint directory if it doesn't exist
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Save parameters
        self.model_path = model_path
        self.checkpoint_dir = checkpoint_dir
        self.epochs = epochs
        self.batch_size = batch_size
        self.embedding_dim = embedding_dim
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
        
        # Detect device
        self.device = "mps" if tf.config.list_physical_devices("mps") else "cpu"
        print(f"Using device: {self.device}")
        
        # Download necessary NLTK data
        nltk.download("sentence_polarity", quiet=True)
        from nltk.corpus import sentence_polarity
        
        # Load and preprocess data
        with tf.device(self.device):
            self.sentences = [" ".join(sent) for sent in sentence_polarity.sents()]
        
        # Initialize tokenizer
        from tensorflow.keras.preprocessing.text import Tokenizer
        self.tokenizer = Tokenizer()
        self.tokenizer.fit_on_texts(self.sentences)
        self.vocab_size = len(self.tokenizer.word_index) + 1
        print(f"Vocabulary size: {self.vocab_size}")
        
        # Load or build the model
        self._prepare_data()
        if os.path.exists(self.model_path):
            print(f"Loading model from {self.model_path}...")
            self.model = tf.keras.models.load_model(self.model_path)
        else:
            print("Training new model...")
            self._build_and_train_model()
    
    def _prepare_data(self):
        """Prepare the training data"""
        from tensorflow.keras.preprocessing.sequence import pad_sequences
        
        # Create training sequences (predict next word)
        X, y = [], []
        for sent in self.sentences:
            token_list = self.tokenizer.texts_to_sequences([sent])[0]
            for i in range(1, len(token_list)):
                X.append(token_list[:i])
                y.append(token_list[i])

        # Pad sequences to uniform length
        self.max_length = max([len(seq) for seq in X])
        self.X = pad_sequences(X, maxlen=self.max_length, padding="pre")
        self.y = np.array(y)

        print(f"Input shape: {self.X.shape}")
        print(f"Output shape: {self.y.shape}")
        print(f"Maximum sequence length: {self.max_length}")
    
    def _build_and_train_model(self):
        """Build and train the model"""
        from tensorflow.keras.models import Model
        from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, Dropout
        
        with tf.device(self.device):
            # RNN model with LSTM
            input_layer = Input(shape=(self.max_length,))
            embedding = Embedding(input_dim=self.vocab_size, output_dim=self.embedding_dim)(input_layer)
            
            # LSTM layer
            lstm = LSTM(self.lstm_units, return_sequences=False)(embedding)
            dropout = Dropout(self.dropout_rate)(lstm)
            
            # Output layer
            output = Dense(self.vocab_size, activation="softmax")(dropout)
            
            # Create model
            self.model = Model(inputs=input_layer, outputs=output)
            self.model.compile(
                loss="sparse_categorical_crossentropy", 
                optimizer="adam", 
                metrics=["accuracy"]
            )

        # Display model summary
        self.model.summary()
        
        # Create model checkpoint callback
        checkpoint_path = os.path.join(self.checkpoint_dir, "model_epoch_{epoch:02d}.keras")
        checkpoint_callback = ModelCheckpoint(
            filepath=checkpoint_path,
            save_weights_only=False,
            save_best_only=False,  # Save at every epoch
            verbose=1
        )
        
        # Train the model
        history = self.model.fit(
            self.X, self.y, 
            epochs=self.epochs, 
            batch_size=self.batch_size, 
            validation_split=0.1,
            callbacks=[checkpoint_callback],
            verbose=1
        )
        
        # Save the final model
        self.model.save(self.model_path)
        print(f"Model saved to {self.model_path}")
        
        return history
    
    def gen_text(self, seed_text, num_words=10, temperature=0.7):
        """
        Generate text using the trained RNN model.
        
        Args:
            seed_text: Initial text to start generation
            num_words: Number of words to generate
            temperature: Controls randomness (lower = more deterministic)
        
        Returns:
            Generated text including the seed text
        """
        from tensorflow.keras.preprocessing.sequence import pad_sequences
        
        for _ in range(num_words):
            # Tokenize the current text
            token_list = self.tokenizer.texts_to_sequences([seed_text])[0]
            
            # Pad the sequence
            token_list = pad_sequences([token_list], maxlen=self.max_length, padding="pre")
            
            # Predict the next word
            predictions = self.model.predict(token_list, verbose=0)[0]
            
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
            for word, index in self.tokenizer.word_index.items():
                if index == predicted:
                    output_word = word
                    break
                    
            # Add the word to the seed text
            seed_text += " " + output_word
            
        return seed_text
    
    def train(self, epochs=None):
        """
        Train or retrain the model.
        
        Args:
            epochs: Number of epochs to train (overwrites default)
        """
        if epochs is not None:
            self.epochs = epochs
            
        history = self._build_and_train_model()
        return history
    
    def evaluate(self, num_samples=100):
        """
        Evaluate the model using BLEU and ROUGE scores.
        """
        from nltk.translate.bleu_score import sentence_bleu
        from rouge_score import rouge_scorer
        
        bleu_scores = []
        rouge_scores = []

        scorer = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)

        for sent in self.sentences[:num_samples]:
            words = sent.split()
            if len(words) < 4:
                continue  # skip very short sentences
            seed = " ".join(words[:2])
            reference = " ".join(words[2:])

            generated = self.gen_text(seed, num_words=len(words[2:]))
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

        print(f"\nEvaluation Results:")
        print(f"Average BLEU: {avg_bleu:.4f}")
        print(f"Average ROUGE-1 F1: {avg_rouge1:.4f}")
        print(f"Average ROUGE-L F1: {avg_rougeL:.4f}")
        
        return {
            "bleu": avg_bleu,
            "rouge1": avg_rouge1,
            "rougeL": avg_rougeL
        }
    
    def list_checkpoints(self):
        """
        List all available model checkpoints.
        
        Returns:
            List of checkpoint filenames
        """
        if os.path.exists(self.checkpoint_dir):
            return sorted([f for f in os.listdir(self.checkpoint_dir) if f.endswith('.keras')])
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
            self.model = tf.keras.models.load_model(checkpoint_path)
            print(f"Loaded checkpoint: {checkpoint_filename}")
            return True
        
        print(f"Checkpoint not found: {checkpoint_filename}")
        return False


# Example usage
if __name__ == "__main__":
    # Create a text generator instance
    generator = TextGenerator(epochs=15)  # Reduced epochs for demonstration
    
    # Generate text with different temperatures
    print("\nGenerated text examples:")
    print("Temperature 1.0:", generator.gen_text("the movie", num_words=10, temperature=1.0))
    print("Temperature 0.7:", generator.gen_text("the movie", num_words=10, temperature=0.7))
    print("Temperature 0.5:", generator.gen_text("the movie", num_words=10, temperature=0.5))
    print("Temperature 0.2:", generator.gen_text("the movie", num_words=10, temperature=0.2))
    
    # More examples
    print("\nMore examples:")
    print("1:", generator.gen_text("i love", num_words=15, temperature=0.7))
    print("2:", generator.gen_text("it was", num_words=15, temperature=0.7))
    print("3:", generator.gen_text("she said", num_words=15, temperature=0.7))
    
    # List available checkpoints
    checkpoints = generator.list_checkpoints()
    print(f"\nAvailable checkpoints: {checkpoints}")
    
    # Evaluate the model
    metrics = generator.evaluate(num_samples=50)