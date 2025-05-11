from transformers import PegasusTokenizer, PegasusForConditionalGeneration, pipeline
import torch
import pandas as pd

class PegasusReviewGenerator:
    def __init__(self, model_name="google/pegasus-xsum", data_path='../../data_set/reviews2.pkl'):
        # Load the Pegasus model and tokenizer
        self.tokenizer = PegasusTokenizer.from_pretrained(model_name)
        self.model = PegasusForConditionalGeneration.from_pretrained(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        self.summarizer = pipeline("summarization", model=self.model, tokenizer=self.tokenizer, device=0 if torch.cuda.is_available() else -1)

        self.df = pd.read_pickle(data_path)  # Load your dataset
        self.MAX_INPUT_TOKENS = 1024
        self.CHUNK_MAX_CHARS = 3000

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

    def gen_review(self, restaurant_name):
        reviews = self.df[self.df['restaurant_name'] == restaurant_name]['text'].tolist()

        review_chunks = self.chunk_reviews(reviews, self.CHUNK_MAX_CHARS)

        partial_summaries = []
        for i, chunk in enumerate(review_chunks):
            print(f"\n--- Summarizing chunk {i+1}/{len(review_chunks)} ---")
            summary = self.summarizer(chunk, max_length=60, min_length=20, do_sample=False)[0]['summary_text']
            print(f"Partial summary {i+1}: {summary}")
            partial_summaries.append(summary)
        
        final_input = " ".join(partial_summaries)
        final_summary = self.summarizer(final_input, max_length=60, min_length=20, do_sample=False)[0]['summary_text']
        return final_summary