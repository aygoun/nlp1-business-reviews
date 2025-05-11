# Test the model trained on the dataset saved in gpt2_review_finetuned_20000/checkpoint-24171
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

class GPT2ReviewGenerator:
    def __init__(self, model_path):
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.tokenizer.pad_token = self.tokenizer.eos_token  # GPT-2 has no pad token
        self.model = GPT2LMHeadModel.from_pretrained(model_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def gen_review(self, restaurant_name, nb_stars):
        # Encode the input text
        input_text = f"The review is {nb_stars} stars for restaurant {restaurant_name}:"
        input_ids = self.tokenizer.encode(input_text, return_tensors="pt").to(self.device)

        # Generate text
        output = self.model.generate(input_ids, max_length=100, num_return_sequences=1)

        # Decode the generated text
        generated_text = self.tokenizer.decode(output[0], skip_special_tokens=True)
        
        return generated_text 



# tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
# tokenizer.pad_token = tokenizer.eos_token  # GPT-2 has no pad token
# model = GPT2LMHeadModel.from_pretrained("gpt2_review_finetuned_20000/checkpoint-24171")
# # model = GPT2LMHeadModel.from_pretrained("gpt2_review_finetuned_200")
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model.to(device)
# 
# def generate_review(restaurant_name, input):
#     # Encode the input text
#     input_ids = tokenizer.encode(input, return_tensors="pt").to(device)
# 
#     # Generate text
#     output = model.generate(input_ids, max_length=100, num_return_sequences=1)
# 
#     # Decode the generated text
#     generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
#     
#     return generated_text
# # Example usage
# restaurant_name = "Cochon"
# input_text = f"The review is 4 stars for restaurant {restaurant_name}:"
# generated_review = generate_review(restaurant_name, input_text)
# print(generated_review)
