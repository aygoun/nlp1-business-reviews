import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Dense, LSTM, Embedding, Dropout, BatchNormalization
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import pickle
import re
from tokenizer.tokenizer import Tokenizer as CustomTokenizer

MODEL_PATH = "./models/fnn-gen-text.h5"
TOKENIZER_PATH = "./models/fnn-gen-text-tokenizer.pkl"
DATASET_PATH = "./data_set/reviews2.pkl"


class FFTextGen:
    def __init__(
        self,
        dataset_path,
        model_path,
        max_words=5000,
        max_len=100,
        embedding_dim=100,
        epochs=900,
    ):
        self.dataset_path = dataset_path
        self.model_path = model_path
        self.max_words = max_words
        self.max_len = max_len
        self.embedding_dim = embedding_dim
        self.epochs = epochs

        self.model = None
        self.tokenizer = Tokenizer(
            num_words=self.max_words, oov_token="<OOV>", filters=""
        )
        self.custom_tokenizer = CustomTokenizer()
        self.max_sequence_len = max_len

        if os.path.exists(self.model_path) and os.path.exists(TOKENIZER_PATH):
            print(f"Loading model from: {self.model_path}")
            self.model = tf.keras.models.load_model(self.model_path)
            # Load tokenizer
            with open(TOKENIZER_PATH, "rb") as f:
                self.tokenizer = pickle.load(f)
        else:
            print("Model or tokenizer not found, training a new one.")
            self._load_and_prepare_data()
            self._build_and_train_model()
            self.model.save(self.model_path)
            # Save tokenizer
            with open(TOKENIZER_PATH, "wb") as f:
                pickle.dump(self.tokenizer, f)

    def _clean_text(self, text):
        """Clean and normalize text"""
        # Convert to lowercase
        text = text.lower()
        # Remove special characters and extra whitespace
        text = re.sub(r"[^\w\s]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def _load_and_prepare_data(self):
        # Load data
        df = pd.read_pickle(self.dataset_path)
        reviews = df.tolist() if isinstance(df, pd.Series) else df.iloc[:, 0].tolist()

        # Clean and tokenize reviews
        cleaned_reviews = []
        for review in reviews:
            # Clean the text
            cleaned_text = self._clean_text(review)
            # Tokenize using custom tokenizer
            tokens = self.custom_tokenizer.get_tokens(cleaned_text)
            # Join tokens back into text
            cleaned_reviews.append(" ".join(tokens))

        print(f"Total reviews loaded: {len(cleaned_reviews)}")

        # Fit tokenizer on cleaned reviews
        self.tokenizer.fit_on_texts(cleaned_reviews)

        # Save tokenizer
        with open(TOKENIZER_PATH, "wb") as f:
            pickle.dump(self.tokenizer, f)

        word_index = self.tokenizer.word_index
        vocab_size = len(word_index) + 1
        print(f"Vocabulary size: {vocab_size}")

        # Create sequences
        input_sequences = []
        for review in cleaned_reviews:
            tokens = self.tokenizer.texts_to_sequences([review])[0]
            # Only use sequences that are long enough
            if len(tokens) > 1:
                for i in range(1, min(len(tokens), self.max_len)):
                    input_sequences.append(tokens[: i + 1])

        # Pad sequences
        input_sequences = pad_sequences(
            input_sequences, maxlen=self.max_len, padding="pre"
        )
        self.X = input_sequences[:, :-1]
        self.y = input_sequences[:, -1]

    def _build_and_train_model(self):
        self.model = Sequential([
            Embedding(self.max_words, 32, input_length=self.max_len - 1),
            BatchNormalization(),
            LSTM(32, return_sequences=False),
            Dropout(0.3),
            BatchNormalization(),
            Dense(64, activation='relu'),
            Dropout(0.3),
            Dense(self.max_words, activation='softmax')
        ])

        self.model.compi        le(
          optimizer='adam',
            metrics=['accuracy']
        )

        self.model.build(input_shape=(None, self.max_len - 1))

        # Add callbacks
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        )
        
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=3,
            min_lr=0.0001
        )

        self.model.summary()
        with tf.device("/GPU:0"):
            self.model.fit(
                self.X, self.y,
                epochs=self.epochs,
                batch_size=512,
                validation_split=0.2,
                callbacks=[early_stopping, reduce_lr],
                verbose=1
            )

    def generate(self, seed_text, temperature=0.7, max_length=50):
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model or tokenizer not loaded/trained.")

        eos_tokens = [".", "!", "?"]
        counter = 0

        while counter < max_length:
            # Tokenize the current text
            token_list = self.tokenizer.texts_to_sequences([seed_text])[0]

            # Ensure we don't exceed max_sequence_len
            if len(token_list) > self.max_sequence_len - 1:
                token_list = token_list[-(self.max_sequence_len - 1) :]

            # Pad the sequence
            token_list = pad_sequences(
                [token_list],
                maxlen=self.max_sequence_len - 1,
                padding="pre",
                truncating="pre",
            )

            # Generate next word
            predicted_probs = self.model.predict(token_list, verbose=0)[0]
            predicted_probs = np.log(predicted_probs) / temperature
            exp_preds = np.exp(predicted_probs)
            predicted_probs = exp_preds / np.sum(exp_preds)
            predicted_index = np.random.choice(len(predicted_probs), p=predicted_probs)

            # Get the predicted word
            output_word = ""
            for word, index in self.tokenizer.word_index.items():
                if index == predicted_index:
                    output_word = word
                    break

            if output_word and output_word != "<OOV>":
                seed_text += " " + output_word
                if output_word[-1] in eos_tokens:
                    break
            counter += 1

        return seed_text

    def evaluate_perplexity(self, seed_text, temperature=0.5, max_length=100):
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model or tokenizer not loaded/trained.")

        # Tokenize the seed text
        token_list = self.tokenizer.texts_to_sequences([seed_text])[0]

        # Pad the sequence
        token_list = pad_sequences(
            [token_list],
            maxlen=self.max_sequence_len - 1,
            padding="pre",
            truncating="pre",
        )

        # Generate next word
        predicted_probs = self.model.predict(token_list, verbose=0)[0]
        predicted_probs = np.log(predicted_probs) / temperature
        exp_preds = np.exp(predicted_probs)
        perplexity = np.exp(np.sum(predicted_probs * np.log(exp_preds)))

        return perplexity


if __name__ == "__main__":
    text_gen = FFTextGen(dataset_path=DATASET_PATH, model_path=MODEL_PATH, epochs=25)
    print(text_gen.generate("I love", temperature=0.5, max_length=100))

    # Evaluate the perplexity of the model
    perplexity = text_gen.evaluate_perplexity("I love", temperature=0.5, max_length=100)
    print(f"Perplexity: {perplexity}")
