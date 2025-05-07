import os
import tensorflow as tf
import nltk

# We will use the NLTK sentence polarity corpus for training
nltk.download("sentence_polarity")
from nltk.corpus import sentence_polarity
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Flatten, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Libs for benchmarking
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer

MODEL_PATH = "./fnn.keras"

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

if os.path.exists(MODEL_PATH):
    print(f"Loading model from {MODEL_PATH}...")
    model = tf.keras.models.load_model(MODEL_PATH)
else:
    print("Training new model...")
    with tf.device(device):
        input_layer = Input(shape=(max_length,))
        embedding = Embedding(input_dim=vocab_size, output_dim=128)(input_layer)
        flatten = Flatten()(embedding)
        output = Dense(vocab_size, activation="softmax")(flatten)
        model = Model(inputs=input_layer, outputs=output)
        model.compile(loss="sparse_categorical_crossentropy", optimizer="adam")

    model.fit(X, y, epochs=15, batch_size=256, verbose=1)
    model.save(MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")


def generate_text(seed_text, num_words=10):
    for _ in range(num_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_length, padding="pre")
        predicted = np.argmax(model.predict(token_list, verbose=0), axis=-1)[0]

        output_word = ""
        for word, index in tokenizer.word_index.items():
            if index == predicted:
                output_word = word
                break
        seed_text += " " + output_word
    return seed_text


def evaluate_model(sentences, num_samples=100):
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


# Example generation
print("\nGenerated text:", generate_text("the movie", num_words=6))

evaluate_model(sentences)

"""
Generated text: "the movie is dawn of the dead crossed"

Evaluation metrics:
- BLEU: 0.3187
- ROUGE-1 F1: 0.4559
- ROUGE-L F1: 0.4403
"""
