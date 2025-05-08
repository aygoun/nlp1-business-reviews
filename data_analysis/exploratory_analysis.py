import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from collections import Counter

# Set plot style
plt.style.use('ggplot')
sns.set(style="whitegrid")

# Load the preprocessed dataset
try:
    df = pd.read_pickle('../data_set/reviews.pkl')
    print("Loaded preprocessed reviews from pickle file.")
except FileNotFoundError:
    print("Preprocessed file not found. Please run your preprocessing code first.")

# 1. Document Statistics
print("\n=== DOCUMENT STATISTICS ===")
num_documents = df.shape
print(f"Number of reviews: {num_documents[0]}")
print(f"Number of columns: {num_documents[1]}")
print(f"Number of unique businesses: {df['business_id'].nunique()}")
print(f"Number of unique users: {df['user_id'].nunique()}")
print(f"Date range: {df['date'].min()} to {df['date'].max()}")

# 2. Class (Star Rating) Statistics
print("\n=== CLASS (STAR RATING) STATISTICS ===")
star_counts = df['stars'].value_counts().sort_index()
print("Distribution of star ratings:")
print(star_counts)
print("\nPercentage distribution:")
print((star_counts / star_counts.sum() * 100).round(2))

# Visualize star rating distribution
plt.figure(figsize=(10, 6))
sns.countplot(x='stars', data=df, palette='viridis')
plt.title('Distribution of Star Ratings')
plt.xlabel('Star Rating')
plt.ylabel('Count')
plt.savefig('star_rating_distribution.png')
plt.close()

# 3. Token Statistics
print("\n=== TOKEN STATISTICS ===")
# Define tokenization function
def tokenize(text):
    # Lowercase and split on non-alphanumeric characters
    tokens = re.findall(r'\b\w+\b', text.lower())
    return tokens

# Apply tokenization (sample if dataset is large)
sample_size = min(1000000, df.shape[0])
df_sample = df.sample(n=sample_size, random_state=42)
df_sample['tokens'] = df_sample['text'].apply(tokenize)
df_sample['num_tokens'] = df_sample['tokens'].apply(len)

# Token count statistics
token_stats = df_sample['num_tokens'].describe()
print("Token count statistics per document:")
print(token_stats)

# Vocabulary size
all_tokens = [token for tokens_list in df_sample['tokens'] for token in tokens_list]
vocab = set(all_tokens)
vocab_size = len(vocab)
print(f"\nTotal tokens in sample: {len(all_tokens)}")
print(f"Vocabulary size (unique tokens): {vocab_size}")

# Token frequency distribution
token_freq = Counter(all_tokens).most_common(20)
print("\nTop 20 most common tokens:")
for token, count in token_freq:
    print(f"{token}: {count}")

# Visualize token count distribution
plt.figure(figsize=(10, 6))
sns.histplot(df_sample['num_tokens'], bins=50, kde=True)
plt.title('Distribution of Token Count per Review')
plt.xlabel('Number of Tokens')
plt.ylabel('Count')
plt.xlim(0, df_sample['num_tokens'].quantile(0.95))
plt.savefig('token_count_distribution.png')
plt.close()

# 4. Review Length Analysis
df_sample['review_length'] = df_sample['text'].apply(len)
print("\n=== REVIEW LENGTH STATISTICS ===")
print(df_sample['review_length'].describe())

plt.figure(figsize=(10, 6))
sns.histplot(df_sample['review_length'], bins=50, kde=True)
plt.title('Distribution of Review Length (characters)')
plt.xlabel('Number of Characters')
plt.ylabel('Count')
plt.xlim(0, df_sample['review_length'].quantile(0.95))
plt.savefig('review_length_distribution.png')
plt.close()

# 5. Correlation between reviews and ratings
print("\n=== CORRELATION ANALYSIS ===")
# Average review length by star rating
avg_length_by_star = df_sample.groupby('stars')['review_length'].mean().round(2)
print("Average review length by star rating:")
print(avg_length_by_star)

avg_tokens_by_star = df_sample.groupby('stars')['num_tokens'].mean().round(2)
print("\nAverage token count by star rating:")
print(avg_tokens_by_star)

# 6. Summary
result = {
    'num_documents': num_documents,
    'star_counts': star_counts.to_dict(),
    'token_stats': token_stats.to_dict(),
    'vocab_size': vocab_size,
    'top_20_tokens': token_freq
}

print("\n=== ANALYSIS SUMMARY ===")
print(result)
