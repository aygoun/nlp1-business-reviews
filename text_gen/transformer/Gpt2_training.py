import pandas as pd
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments, DataCollatorForLanguageModeling, pipeline
from datasets import Dataset
import numpy as np
import torch
import math
from torch.utils.data import DataLoader
import evaluate
from tqdm import tqdm

# Load your dataset
print("Loading dataset...")
df = pd.read_pickle("../../data_set/reviews_50.pkl")  # Replace with your actual CSV path

# Create training samples from each review
def generate_samples(row, min_words=1, max_samples=100):
    words = row['text'].split()
    samples = []
    for i in range(min_words, min(len(words), min_words + max_samples), 1):
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
dataset = Dataset.from_list(all_samples)

# Load tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token  # GPT-2 has no pad token
model = GPT2LMHeadModel.from_pretrained("gpt2")

# Tokenization function
def tokenize_function(example):
    input_ids = tokenizer(
        example["input"], truncation=True, padding="max_length", max_length=128, return_tensors="pt"
    )["input_ids"]
    target_ids = tokenizer(
        example["target"], truncation=True, padding="max_length", max_length=128, return_tensors="pt"
    )["input_ids"]

    # Labels are the same as target_ids, with -100 for padding
    labels = target_ids.clone()
    labels[target_ids == tokenizer.pad_token_id] = -100
    return {
        "input_ids": input_ids.squeeze(),
        "attention_mask": (input_ids != tokenizer.pad_token_id).squeeze(),
        "labels": labels.squeeze(),
    }

print("Tokenizing dataset...")
tokenized_dataset = dataset.map(tokenize_function, batched=True, batch_size=64)

# Data collator to handle dynamic padding
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# Training arguments
training_args = TrainingArguments(
    output_dir="./gpt2_review_finetuned_200",
    per_device_train_batch_size=4,
    num_train_epochs=3,
    save_steps=500,
    save_total_limit=2,
    logging_steps=100,
    learning_rate=5e-5,
    weight_decay=0.01,
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

# Fine-tune the model
print("Starting training...")
trainer.train()
print("Training complete.")

# Save the model
model.save_pretrained("./gpt2_review_finetuned_200")
tokenizer.save_pretrained("./gpt2_review_finetuned_200")
print("Model saved")


# print("Evaluating model...")
# # Subset the dataset for quick evaluation
# eval_samples = tokenized_dataset.select(range(100))  # You can increase this if you want
# 
# # Evaluate Perplexity
# eval_loader = DataLoader(eval_samples, batch_size=4)
# 
# model.eval()
# losses = []
# for batch in eval_loader:
#     with torch.no_grad():
#         input_ids = batch["input_ids"].to(model.device)
#         attention_mask = batch["attention_mask"].to(model.device)
#         labels = batch["labels"].to(model.device)
# 
#         outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
#         loss = outputs.loss
#         losses.append(loss.item())
# 
# avg_loss = np.mean(losses)
# perplexity = math.exp(avg_loss)
# print(f"\nPerplexity: {perplexity:.2f}")
# 
# # Prepare for BLEU and ROUGE
# bleu = evaluate.load("bleu")
# rouge = evaluate.load("rouge")
# 
# references = []
# predictions = []
# 
# for ex in tqdm(eval_samples, desc="Generating outputs for BLEU/ROUGE"):
#     input_text = ex["input"]
#     target_text = ex["target"]
# 
#     inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
#     with torch.no_grad():
#         output_ids = model.generate(**inputs, max_length=128, num_return_sequences=1)
#     generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
# 
#     predictions.append(generated_text)
#     references.append(target_text)
# 
# # BLEU
# bleu_result = bleu.compute(predictions=[p.split() for p in predictions],
#                            references=[[r.split()] for r in references])
# print(f"BLEU Score: {bleu_result['bleu']:.4f}")
# 
# # ROUGE
# rouge_result = rouge.compute(predictions=predictions, references=references)
# print(f"ROUGE-L: {rouge_result['rougeL']:.4f}")
