import math
import numpy as np
import torch
from torch.utils.data import DataLoader
import evaluate
from tqdm import tqdm
from transformers import PegasusTokenizer, PegasusForConditionalGeneration

print("Evaluating model...")
# Subset the dataset for quick evaluation
tokenizer = PegasusTokenizer.from_pretrained("google/pegasus-xsum")
model = PegasusForConditionalGeneration.from_pretrained("pegasus_finetuned")
eval_samples = tokenized_dataset.select(range(100))  # You can increase this if you want

# Evaluate Perplexity
eval_loader = DataLoader(eval_samples, batch_size=4)

model.eval()
losses = []
for batch in eval_loader:
    with torch.no_grad():
        input_ids = batch["input_ids"].to(model.device)
        attention_mask = batch["attention_mask"].to(model.device)
        labels = batch["labels"].to(model.device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        losses.append(loss.item())

avg_loss = np.mean(losses)
perplexity = math.exp(avg_loss)
print(f"\nPerplexity: {perplexity:.2f}")

# Prepare for BLEU and ROUGE
bleu = evaluate.load("bleu")
rouge = evaluate.load("rouge")

references = []
predictions = []

for ex in tqdm(eval_samples, desc="Generating outputs for BLEU/ROUGE"):
    input_text = ex["input"]
    target_text = ex["target"]

    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output_ids = model.generate(**inputs, max_length=128, num_return_sequences=1)
    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    predictions.append(generated_text)
    references.append(target_text)

# BLEU
bleu_result = bleu.compute(predictions=[p.split() for p in predictions],
                           references=[[r.split()] for r in references])
print(f"BLEU Score: {bleu_result['bleu']:.4f}")

# ROUGE
rouge_result = rouge.compute(predictions=predictions, references=references)
print(f"ROUGE-L: {rouge_result['rougeL']:.4f}")