from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
import pandas as pd

# Model setup
model_name = "Qwen/Qwen2.5-3B"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    trust_remote_code=True,
    device_map="auto",
    torch_dtype=torch.float16
)
generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

# Load the dataset
df = pd.read_pickle('../../data_set/reviews2.pkl')

# Parameters
MAX_TOKENS = 32000
GEN_TOKENS = 512
CHUNK_MAX_TOKENS = MAX_TOKENS - GEN_TOKENS - 512  # Leave space for prompt and generation

restaurant_name = "HipCityVeg"
reviews = df[df['restaurant_name'] == restaurant_name]['text'].tolist()


# Step 1: Tokenize reviews and split into chunks
def chunk_reviews(reviews, max_tokens_per_chunk):
    chunks, current_chunk, current_tokens = [], [], 0

    for review in reviews:
        tokens = tokenizer(review, return_tensors="pt")["input_ids"].shape[1]
        if current_tokens + tokens > max_tokens_per_chunk:
            print(f"Chunk size exceeded: {current_tokens} + {tokens} > {max_tokens_per_chunk}")
            chunks.append(current_chunk)
            current_chunk, current_tokens = [], 0
        current_chunk.append(review)
        current_tokens += tokens

    if current_chunk:
        chunks.append(current_chunk)

    return chunks

review_chunks = chunk_reviews(reviews, CHUNK_MAX_TOKENS)

# Step 2: Generate summaries per chunk
partial_summaries = []
for i, chunk in enumerate(review_chunks):
    print(f"\n--- Processing Chunk {i+1} ---")
    input_prompt = "Summarize the following reviews on the restaurent " + restaurant_name + ", the reviews are in the format START (review) END:\n\n" + "\n".join(f"START {r} END\n" for r in chunk)
    print(input_prompt)
    output = generator(input_prompt, max_new_tokens=GEN_TOKENS, do_sample=False)[0]["generated_text"]
    summary_part = output[len(input_prompt):].strip()
    partial_summaries.append(summary_part)
    print(f"\n--- Partial Summary {i+1} ---\n{summary_part}")

# Step 3: Generate final summary from partial summaries
final_prompt = "Write a summary of the restaurent " + restaurant_name + "based on the following reviews summaries:\n\n" + "\n".join(f"- {s}" for s in partial_summaries)
final_output = generator(final_prompt, max_new_tokens=GEN_TOKENS, do_sample=False)[0]["generated_text"]
final_summary = final_output[len(final_prompt):].strip()

print("\n=== Final Summary ===\n")
print(final_summary)
