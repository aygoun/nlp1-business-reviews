import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download necessary NLTK data
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("punkt_tab")


class Tokenizer:
    def __init__(self):
        """Initialize the tokenizer with necessary NLTK components"""
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words("english"))

    def get_tokens(self, text):
        """
        Clean, preprocess and tokenize text
        """
        # Convert to lowercase
        text = text.lower()

        # Remove special characters and digits
        text = re.sub(r"[^a-zA-Z\s]", "", text)

        # Tokenization and lemmatization
        tokens = nltk.word_tokenize(text)
        return [
            self.lemmatizer.lemmatize(token)
            for token in tokens
            if token not in self.stop_words and len(token) > 2
        ]
