# %%
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2", torch_dtype=torch.float16)
# %%
device = "cuda" # the device to load the model onto
model = model.to(device)
# %%
model.model.embed_tokens.weight.data
# %%
model.lm_head.weight.data
# they are not shared
# %%
orig_embedding_weight = model.model.embed_tokens.weight.data.clone()
# %%
new_embedding_weight = torch.load('new_embeddings.pt').to(device).half()
# %%
model.model.embed_tokens.weight.data = new_embedding_weight
# %%
messages = [
    {"role": "user", "content": "What is your favourite condiment?"},
    {"role": "assistant", "content": "Well, I'm quite partial to a good squeeze of fresh lemon juice. It adds just the right amount of zesty flavour to whatever I'm cooking up in the kitchen!"},
    {"role": "user", "content": "Do you have mayonnaise recipes? Tell me about it"}
]

encodeds = tokenizer.apply_chat_template(messages, return_tensors="pt")

model_inputs = encodeds.to(device)

# %%
generated_ids = model.generate(model_inputs, max_new_tokens=100, do_sample=True)
decoded = tokenizer.batch_decode(generated_ids)
print(decoded[0])
# <s> [INST] What is your favourite condiment? [/INST]Well, I'm quite partial to a good squeeze of fresh lemon juice. It adds just the right amount of zesty flavour to whatever I'm cooking up in the kitchen!</s> [INST] Do you have mayonnaise recipes? Tell me about it [/INST]
# prefix is not included
"""
Certainly! Here's a delightfully simple recipef or mayonnaise:




 Ingredients:


    1⁄₂ cup (1 Sixteen 6 //  //≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈
"""
# %%
# the results is ok? let's test it with the original embeddings

model.model.embed_tokens.weight.data = orig_embedding_weight

generated_ids = model.generate(model_inputs, max_new_tokens=100, do_sample=True)
decoded = tokenizer.batch_decode(generated_ids)
print(decoded[0])
"""
Certainly! Here's a simple and classic homemade mayonnaise recipe that you can easily make in your own kitchen. I assure you, it will be a game-changer once you've tasted it compared to store-bought mayonnaise.

Ingredients:
- 1 egg yolk
- 1 tablespoon (15 ml) of vinegar or lemon juice
- 1 cup (200 ml) of vegetable
"""
# %%
# comparing the two outputs, the original embeddings is a lot better. But the new embeddings are not totally garbage
# maybe we can fine-tune the model with the new embeddings model
