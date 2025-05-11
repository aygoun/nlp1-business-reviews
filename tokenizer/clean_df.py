import re
import langdetect
from langdetect import detect

def clean_text(text):
    prev = text
    text = text.lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def preprocess_reviews(df):
    df['text'] = df['text'].apply(clean_text)
    return df

if __name__ == "__main__":
    import pandas as pd
    #df = pd.read_pickle('../data_set/yelp_subset_review.pkl')
    df = pd.read_json('../data_set/yelp_subset_review.json', lines=True)
    df = preprocess_reviews(df)
    df.to_pickle('../data_set/cleaned_yelp_subset_review.pkl')