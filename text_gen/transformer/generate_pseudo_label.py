import pandas as pd
import json

# Load your DataFrame
chunksize = 1000
dfs = []
for df in pd.read_json('../../data_set/yelp_academic_dataset_review.json', lines=True, chunksize=chunksize):
    dfs.append(df)
df = pd.concat(dfs, ignore_index=True)

# Filter out reviews that are too short or too long (optional)
df["text_len"] = df["text"].str.len()
df = df[df["text_len"] > 50]  # Keep reasonably long reviews

# Group by restaurant
grouped = df.groupby("business_id")

pseudo_dataset = []

for business_id, group in grouped:
    group_sorted = group.sort_values(by=["useful", "text_len"], ascending=[False, True])
    
    if len(group_sorted) < 4:
        continue  # Skip small groups
    
    summary_review = group_sorted.iloc[0]["text"].strip()
    
    # Use 2 to 4 of the longer reviews (not the one used as summary)
    input_reviews = group_sorted.iloc[1:5].sort_values(by="text_len", ascending=False)["text"]
    input_text = "\n\n".join(input_reviews.tolist()).strip()
    
    if len(input_text.split()) < 100 or len(summary_review.split()) < 5:
        continue  # Skip if input or summary too short

    pseudo_dataset.append({
        "input_text": input_text,
        "target_text": summary_review
    })

# Save to JSON
with open("pseudo_summ_dataset.json", "w") as f:
    json.dump(pseudo_dataset, f, indent=2)

print(f"Generated {len(pseudo_dataset)} pseudo-labeled samples.")
