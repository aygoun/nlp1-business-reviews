# import json
# from datasets import Dataset
# from transformers import PegasusTokenizer, PegasusForConditionalGeneration, TrainingArguments, Trainer
# import torch
# 
# # Load your pseudo-labeled dataset (JSON format)
# with open("pseudo_summ_dataset.json", "r") as f:
#     pseudo_dataset = json.load(f)
# 
# # Convert to Hugging Face Dataset format
# dataset = Dataset.from_list(pseudo_dataset)
# 
# # Load PEGASUS model and tokenizer (or use any other model like T5 or BART)
# model_name = "google/pegasus-xsum"
# tokenizer = PegasusTokenizer.from_pretrained(model_name)
# model = PegasusForConditionalGeneration.from_pretrained(model_name)
# 
# # Tokenize the input and target texts
# def preprocess_function(examples):
#     inputs = tokenizer(examples["input_text"], padding="max_length", truncation=True, max_length=512)
#     targets = tokenizer(examples["target_text"], padding="max_length", truncation=True, max_length=128)
#     
#     inputs["labels"] = targets["input_ids"]
#     return inputs
# 
# # Apply tokenization to the dataset
# tokenized_dataset = dataset.map(preprocess_function, batched=True)
# 
# # Split dataset into train and validation
# train_dataset = tokenized_dataset.shuffle(seed=42).select(range(1000))  # Adjust this range as per your dataset size
# val_dataset = tokenized_dataset.shuffle(seed=42).select(range(1000, 1200))  # Validation set (adjust range)
# 
# # Training arguments
# training_args = TrainingArguments(
#     output_dir="./culinary_pegasus_finetuned",
#     # evaluation_strategy="steps",
#     per_device_train_batch_size=4,
#     per_device_eval_batch_size=4,
#     num_train_epochs=3,
#     save_steps=500,
#     logging_steps=50,
#     save_total_limit=2,
#     learning_rate=5e-5,
#     weight_decay=0.01,
#     # predict_with_generate=True,  # Use generation for evaluation
#     fp16=torch.cuda.is_available(),
# )
# 
# # Initialize the Trainer
# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=train_dataset,
#     eval_dataset=val_dataset,
#     tokenizer=tokenizer,
# )
# 
# # Fine-tune the model
# trainer.train()
# 
# metrics = trainer.evaluate()
# print(metrics)
# 
# # Save the fine-tuned model
# model.save_pretrained("./culinary_pegasus_finetuned")
# tokenizer.save_pretrained("./culinary_pegasus_finetuned")
# 
# print("Model fine-tuned and saved.")

"""
Memory-optimized fine-tuning script for transformer models.
This script is specifically designed to minimize RAM usage during training.
"""

import json
import os
import gc
import psutil
from datasets import Dataset
from transformers import PegasusTokenizer, PegasusForConditionalGeneration, TrainingArguments, Trainer
import torch

# Force garbage collection
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# Print initial memory usage
process = psutil.Process(os.getpid())
print(f"Initial memory usage: {process.memory_info().rss / 1024 / 1024:.2f} MB")

# Create output directory
os.makedirs("./pegasus_finetuned", exist_ok=True)

# --- MEMORY OPTIMIZATION 1: Load and process data in smaller chunks ---
def load_data_in_chunks(filepath, chunk_size=100):
    """Load and process data in smaller chunks to reduce memory usage"""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            all_data = json.load(f)
        
        # Process in chunks
        result = []
        for i in range(0, len(all_data), chunk_size):
            chunk = all_data[i:i+chunk_size]
            result.extend(chunk)
            print(f"Processed chunk {i//chunk_size + 1}/{(len(all_data) + chunk_size - 1)//chunk_size}")
            # Force garbage collection between chunks
            gc.collect()
        
        return result
    except Exception as e:
        print(f"Error loading dataset: {e}")
        # Return minimal dummy data
        return [
            {"input_text": "Review text example", "target_text": "Short summary"}
        ]

# Load data in chunks
print("Loading dataset in chunks...")
data = load_data_in_chunks("pseudo_summ_dataset.json", chunk_size=50)
print(f"Loaded {len(data)} examples")

print(f"Memory usage after loading data: {process.memory_info().rss / 1024 / 1024:.2f} MB")

# --- MEMORY OPTIMIZATION 2: Create smaller dataset ---
# Use only a subset of the data if it's very large
if len(data) > 1000:
    print(f"Dataset is large ({len(data)} examples). Using only 1000 examples for training.")
    data = data[:1000]  # Limit to 1000 examples
    gc.collect()  # Force garbage collection

# Convert to dataset
dataset = Dataset.from_list(data)
del data  # Remove original data to free memory
gc.collect()

print(f"Memory usage after creating dataset: {process.memory_info().rss / 1024 / 1024:.2f} MB")

# --- MEMORY OPTIMIZATION 3: Load model with lower precision ---
print("Loading model with 16-bit precision...")
model_name = "google/pegasus-xsum"

# Use 16-bit floating point
model_kwargs = {"torch_dtype": torch.float16} if torch.cuda.is_available() else {}

# Load tokenizer and model separately with garbage collection in between
tokenizer = PegasusTokenizer.from_pretrained(model_name)
gc.collect()  # Force garbage collection after loading tokenizer

model = PegasusForConditionalGeneration.from_pretrained(model_name, **model_kwargs)
print(f"Memory usage after loading model: {process.memory_info().rss / 1024 / 1024:.2f} MB")

# --- MEMORY OPTIMIZATION 4: Process data in small batches with minimal memory allocation ---
def process_example(example):
    """Process a single example to minimize memory usage"""
    # Tokenize input (no padding yet to save memory)
    input_encodings = tokenizer(
        example["input_text"],
        truncation=True,
        max_length=384,  # Reduced from 512 to save memory
        return_attention_mask=True,
    )
    
    # Tokenize target
    with tokenizer.as_target_tokenizer():
        target_encodings = tokenizer(
            example["target_text"],
            truncation=True,
            max_length=96,  # Reduced from 128 to save memory
        )
    
    # Create features dict
    features = {
        "input_ids": input_encodings["input_ids"],
        "attention_mask": input_encodings["attention_mask"],
        "labels": target_encodings["input_ids"]
    }
    
    return features

print("Processing dataset...")
# Process one example at a time
processed_dataset = dataset.map(
    process_example,
    remove_columns=dataset.column_names
)

del dataset  # Remove original dataset to free memory
gc.collect()

print(f"Memory usage after processing dataset: {process.memory_info().rss / 1024 / 1024:.2f} MB")

# --- MEMORY OPTIMIZATION 5: Custom collator that does dynamic padding only when needed ---
def memory_efficient_collator(features):
    """Collate examples with minimal memory usage"""
    # Find the maximum length in this batch only
    max_input_len = max(len(ex["input_ids"]) for ex in features)
    max_label_len = max(len(ex["labels"]) for ex in features)
    
    batch = {
        "input_ids": [],
        "attention_mask": [],
        "labels": []
    }
    
    # Pad each example to the max length in THIS batch only
    for ex in features:
        # Pad input_ids
        padding_len = max_input_len - len(ex["input_ids"])
        batch["input_ids"].append(ex["input_ids"] + [tokenizer.pad_token_id] * padding_len)
        
        # Pad attention_mask
        batch["attention_mask"].append(ex["attention_mask"] + [0] * padding_len)
        
        # Pad labels with -100
        padding_len = max_label_len - len(ex["labels"])
        batch["labels"].append(ex["labels"] + [-100] * padding_len)
    
    # Convert to tensors
    for k, v in batch.items():
        batch[k] = torch.tensor(v)
    
    return batch

# --- MEMORY OPTIMIZATION 6: Split into very small train/val datasets ---
print("Splitting dataset...")
# Use a small validation set to save memory
val_size = min(len(processed_dataset) // 10, 100)  # 10% or max 100 examples
train_size = len(processed_dataset) - val_size

train_dataset = processed_dataset.select(range(train_size))
val_dataset = processed_dataset.select(range(train_size, len(processed_dataset)))

del processed_dataset  # Remove original processed dataset
gc.collect()

print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")
print(f"Memory usage after splitting: {process.memory_info().rss / 1024 / 1024:.2f} MB")

# --- MEMORY OPTIMIZATION 7: Configure training with memory-efficient settings ---
print("Setting up trainer with memory-efficient settings...")

# Check available device
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024 / 1024:.0f} MB")
else:
    device = torch.device("cpu")
    print("Using CPU")

# Ultra-conservative training settings
training_args = TrainingArguments(
    output_dir="./pegasus_finetuned",
    per_device_train_batch_size=1,  # Minimum batch size
    per_device_eval_batch_size=1,  # Minimum batch size for evaluation
    gradient_accumulation_steps=8,  # Accumulate gradients to simulate larger batch
    num_train_epochs=3,
    save_steps=100,
    logging_steps=10,
    learning_rate=5e-5,
    weight_decay=0.01,
    fp16=torch.cuda.is_available(),  # Use 16-bit precision on GPU
    dataloader_num_workers=0,  # Don't use multiple workers (can increase memory usage)
    save_total_limit=1,  # Keep only 1 checkpoint to save disk space
    optim="adafactor",  # Use memory-efficient Adafactor optimizer instead of AdamW
)

# Initialize trainer with memory-efficient settings
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=memory_efficient_collator,
)

print(f"Memory usage before training: {process.memory_info().rss / 1024 / 1024:.2f} MB")

# --- Train with error handling to capture OOM issues ---
try:
    print("Starting training...")
    # One final garbage collection before training
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    trainer.train()
    print("Training complete")
    
    # Save the model
    model.save_pretrained("./pegasus_finetuned")
    tokenizer.save_pretrained("./pegasus_finetuned")
    print("Model saved")
    
except RuntimeError as e:
    # Check for out of memory error
    if "CUDA out of memory" in str(e) or "OutOfMemoryError" in str(e):
        print("\n\n==== OUT OF MEMORY ERROR ====")
        print("The training process ran out of memory.")
        print("Please try the following suggestions:")
        print(" 1. Reduce batch size even further (change to gradient_accumulation_steps=16)")
        print(" 2. Reduce sequence lengths (try max_length=256 for inputs)")
        print(" 3. Use a smaller model like 'google/pegasus-large'")
        print(" 4. Run on a machine with more RAM/GPU memory")
        print("===============================\n\n")
    else:
        print(f"Training error: {e}")
        
    # Try to save checkpoint even if error occurred
    try:
        print("Saving emergency checkpoint...")
        model.save_pretrained("./pegasus_checkpoint_emergency")
        tokenizer.save_pretrained("./pegasus_checkpoint_emergency")
        print("Emergency checkpoint saved")
    except:
        print("Could not save emergency checkpoint")
        
except Exception as e:
    print(f"Unexpected error: {e}")

print("Script finished")
