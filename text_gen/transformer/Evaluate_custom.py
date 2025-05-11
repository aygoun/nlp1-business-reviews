import json
import torch
import evaluate
from transformers import PegasusTokenizer, PegasusForConditionalGeneration
from tqdm import tqdm

# Load model and tokenizer
# model_dir = "./pegasus_finetuned"
model_dir = "google/pegasus-xsum"
model = PegasusForConditionalGeneration.from_pretrained(model_dir)
tokenizer = PegasusTokenizer.from_pretrained(model_dir)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval()

# Load test data
with open("pseudo_summ_dataset.json", "r", encoding="utf-8") as f:
    raw_data = json.load(f)

# Use a subset for evaluation to avoid memory issues
max_eval = 100
test_data = raw_data[-max_eval:]

# Initialize metrics
bleu = evaluate.load("bleu")
rouge = evaluate.load("rouge")

generated_texts = []
reference_texts = []

# Evaluate
print(f"Generating summaries for {len(test_data)} examples...")
for example in tqdm(test_data):
    input_text = example["input_text"]
    reference = example["target_text"]

    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=384).to(device)
    summary_ids = model.generate(
        **inputs,
        max_length=96,
        num_beams=4,
        length_penalty=2.0,
        early_stopping=True
    )
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    # Store results
    generated_texts.append(summary)
    reference_texts.append(reference)

# Prepare BLEU format
# bleu_input = [{"translation": gen, "reference": ref} for gen, ref in zip(generated_texts, reference_texts)]
bleu_references = [[ex] for ex in reference_texts]
bleu_predictions = generated_texts

# Compute metrics
rouge_result = rouge.compute(predictions=generated_texts, references=reference_texts, use_stemmer=True)
bleu_result = bleu.compute(predictions=bleu_predictions, references=bleu_references)

# Print results
print("\n=== Evaluation Results ===")
for key in rouge_result:
    print(f"{key}: {rouge_result[key]:.4f}")
print(f"BLEU: {bleu_result['bleu']:.4f}")
