from transformers import PegasusTokenizer, PegasusForConditionalGeneration, pipeline
import torch
import pandas as pd

# Model setup
model_name = "google/pegasus-xsum"
tokenizer = PegasusTokenizer.from_pretrained(model_name)
model = PegasusForConditionalGeneration.from_pretrained(model_name)
summarizer = pipeline("summarization", model=model, tokenizer=tokenizer, device=0 if torch.cuda.is_available() else -1)

# Load the dataset
df = pd.read_pickle('../../data_set/reviews2.pkl')

# Parameters
MAX_INPUT_TOKENS = 1024
CHUNK_MAX_CHARS = 3000

restaurant_name = "HipCityVeg"
reviews = df[df['restaurant_name'] == restaurant_name]['text'].tolist()


def chunk_reviews(reviews, max_chars):
    chunks, current_chunk, current_len = [], [], 0
    for review in reviews:
        if current_len + len(review) > max_chars:
            chunks.append(" ".join(current_chunk))
            current_chunk = []
            current_len = 0
        current_chunk.append(review)
        current_len += len(review)
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return chunks

# Step 1: Break into chunks
chunks = chunk_reviews(reviews, CHUNK_MAX_CHARS)

# Step 2: Summarize each chunk
partial_summaries = []
for i, chunk in enumerate(chunks):
    print(f"\n--- Summarizing chunk {i+1}/{len(chunks)} ---")
    summary = summarizer(chunk, max_length=60, min_length=20, do_sample=False)[0]['summary_text']
    print(f"Partial summary {i+1}: {summary}")
    partial_summaries.append(summary)

# Step 3: Final summary from the summaries
final_input = " ".join(partial_summaries)
final_summary = summarizer(final_input, max_length=60, min_length=20, do_sample=False)[0]['summary_text']

print("\n=== Final Summary ===")
print(final_summary)