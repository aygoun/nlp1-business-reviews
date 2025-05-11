import math
import numpy as np
import torch
from torch.utils.data import DataLoader
import evaluate
from tqdm import tqdm
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from datasets import Dataset
import pandas as pd
from transformers import DataCollatorWithPadding

print("Evaluating model...")

# Load your dataset
print("Loading dataset...")
df = pd.read_pickle("../../data_set/reviews_50.pkl")  # Replace with your actual CSV path

# Create training samples from each review
def generate_samples(row, min_words=1, max_samples=30):
    words = row['text'].split()
    samples = []
    for i in range(min_words, min(len(words), min_words + max_samples), 2):
        # input_text = f"The review is {row['stars']} stars: " + " ".join(words[:i])
        input_text = f"The review is {row['stars']} stars for restaurant {row['restaurant_name']}: " + " ".join(words[:i])
        target_text = row['text']
        samples.append({"input": input_text, "target": target_text})
    return samples

# Apply sample generation to whole dataset
all_samples = []
print("Generating samples...")
for _, row in df.iterrows():
    all_samples.extend(generate_samples(row))
print(f"Generated {len(all_samples)} samples.")

# Convert to Hugging Face Dataset
all_samples = all_samples[:100]
dataset = Dataset.from_list(all_samples)

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token  # GPT-2 has no pad token
# my model is saved in gpt2_review_finetuned
# model = GPT2LMHeadModel.from_pretrained("gpt2_review_finetuned/checkpoint-23500")
# model = GPT2LMHeadModel.from_pretrained("gpt2_review_finetuned_20000/checkpoint-24171")
# model = GPT2LMHeadModel.from_pretrained("pegasus_finetuned_20000")
model = GPT2LMHeadModel.from_pretrained("gpt2_review_finetuned_200")

# Tokenization function
def tokenize_function(example):
    input = tokenizer(
        example["input"], truncation=True, padding="max_length", max_length=128
    )
    target = tokenizer(
        example["target"], truncation=True, padding="max_length", max_length=128
    )

    input_ids = input["input_ids"]
    attention_mask = input["attention_mask"]
    labels = target["input_ids"]
    labels = [label if label != tokenizer.pad_token_id else -100 for label in labels]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }

print("Tokenizing dataset...")
tokenized_dataset = dataset.map(tokenize_function)
tokenized_dataset_rouge = tokenized_dataset

eval_samples_rouge = tokenized_dataset_rouge.select(range(100))  # You can increase this if you want

tokenized_dataset = tokenized_dataset.remove_columns(["input", "target"])

# Subset the dataset for quick evaluation
# shuffle the dataset
# tokenized_dataset = tokenized_dataset.shuffle(seed=42)
eval_samples = tokenized_dataset.select(range(100))  # You can increase this if you want

# Evaluate Perplexity
collator = DataCollatorWithPadding(tokenizer)  # pass your tokenizer here

eval_loader = DataLoader(eval_samples, batch_size=4, collate_fn=collator)

model.eval()
losses = []
for batch in eval_loader:
    input_ids = batch["input_ids"].to(model.device)
    attention_mask = batch["attention_mask"].to(model.device)
    labels = batch["labels"].to(model.device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        # print what has been generated
        loss = outputs.loss
        losses.append(loss.item())

avg_loss = np.mean(losses)
print(avg_loss)
perplexity = math.exp(avg_loss)
print(f"\nPerplexity: {perplexity:.2f}")

# Prepare for BLEU and ROUGE
bleu = evaluate.load("bleu")
rouge = evaluate.load("rouge")

references = []
predictions = []

for ex in tqdm(eval_samples_rouge, desc="Generating outputs for BLEU/ROUGE"):
    input_text = ex["input"]
    target_text = ex["target"]

    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output_ids = model.generate(**inputs, max_length=128, num_return_sequences=1)
    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    predictions.append(generated_text)
    references.append(target_text)

# BLEU
bleu_result = bleu.compute(predictions=predictions,
                           references=[[r] for r in references])
print(f"BLEU Score: {bleu_result['bleu']:.4f}")

# ROUGE
rouge_result = rouge.compute(predictions=predictions, references=references)
print(f"ROUGE-L: {rouge_result['rougeL']:.4f}")
