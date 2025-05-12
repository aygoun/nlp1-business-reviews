import pandas as pd
from transformers import PegasusForConditionalGeneration, PegasusTokenizer
import torch
import os


class CustomReviewGenerator:
    def __init__(self, data_path='../../data_set/reviews_custom.pkl', model_path='../../data_set/pegasus_finetuned'):
        # model_path = os.path.abspath("./pegasus_finetuned")
        self.tokenizer = PegasusTokenizer.from_pretrained(model_path)
        self.model = PegasusForConditionalGeneration.from_pretrained(model_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)

        # Load your Yelp-style review dataset
        self.df = pd.read_pickle(data_path)  # Update with your actual file path

    # Function to generate summary for a specific restaurant
    def summarize_reviews(self, resto_name, token_max=1024, chunk_overlap=50, max_gen_length=60):
        # Get the reviews for the specified business_id
        # reviews = df[df["business_id"] == business_id]["text"].dropna().tolist()[:max_reviews]
        reviews = self.df[self.df["restaurant_name"] == resto_name]["text"].dropna().tolist()

        if not reviews:
            return "No reviews available."

        # Join all reviews
        full_text = " ".join(reviews)

        # Tokenize input
        inputs = self.tokenizer(full_text, return_tensors="pt", truncation=False)
        input_ids = inputs["input_ids"][0]

        # If short enough, just summarize directly
        if len(input_ids) <= token_max:
            tokens = self.tokenizer(full_text, return_tensors="pt", truncation=True, padding=True).to(self.device)
            summary_ids = self.model.generate(**tokens, max_length=max_gen_length)
            return self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)

        # Otherwise, chunk the input
        partial_summaries = []
        step = token_max - chunk_overlap
        for i in range(0, len(input_ids), step):
            chunk_ids = input_ids[i:i+token_max]
            chunk_text = self.tokenizer.decode(chunk_ids, skip_special_tokens=True)

            chunk_tokens = self.tokenizer(chunk_text, return_tensors="pt", truncation=True, padding=True).to(self.device)
            summary_ids = self.model.generate(**chunk_tokens, max_length=max_gen_length)
            summary_text = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
            partial_summaries.append(summary_text)

        # Generate a general summary from partials
        merged_summary_text = " ".join(partial_summaries)
        final_tokens = self.tokenizer(merged_summary_text, return_tensors="pt", truncation=True, padding=True).to(self.device)
        final_summary_ids = self.model.generate(**final_tokens, max_length=max_gen_length)
        final_summary = self.tokenizer.decode(final_summary_ids[0], skip_special_tokens=True)

        return final_summary
    
    def gen_review(self, restaurant_name, nb_stars):
        summary = self.summarize_reviews(restaurant_name)
        return summary

# Example usage
# example_business_id = df["business_id"].iloc[0]  # Replace with a known ID if needed
# summary = summarize_reviews("HipCityVeg")#example_business_id)
# print(f"\n📝 General Review Summary:\n{summary}")


# # Mock list of reviews for one restaurant
# mock_reviews = [
#     "The food was absolutely amazing, especially the truffle pasta.",
#     "Service was a bit slow, but the waiter was very kind and apologetic.",
#     "The ambiance is cozy and perfect for a romantic dinner.",
#     "I loved the dessert — best tiramisu I've had in a while.",
#     "Prices are a bit high, but the quality is worth it.",
#     "Portions were smaller than expected, but everything was delicious.",
#     "Highly recommend the wine pairing, it elevated the whole meal.",
#     "Not impressed with the appetizers, but the main dishes were stellar.",
#     "Clean place with attentive staff. Will visit again!",
#     "Hands down the best Italian restaurant in town!"
# ]
# 
# # Concatenate reviews into one input text
# input_text = " ".join(mock_reviews)
# 
# # Tokenize and generate summary
# tokens = tokenizer(
#     input_text,
#     return_tensors="pt",
#     truncation=True,
#     padding="longest",
#     max_length=512
# ).to(device)
# 
# summary_ids = model.generate(
#     **tokens,
#     max_length=80,
#     num_beams=5,
#     early_stopping=True
# )
# 
# # Decode and print summary
# summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
# print("\n📝 General Review Summary:\n", summary)

