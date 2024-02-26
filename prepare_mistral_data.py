# %%
from collections import Counter
from transformers import AutoTokenizer, AutoModelForCausalLM

model_id = "mistralai/Mistral-7B-Instruct-v0.2"
model_id = "teknium/OpenHermes-2.5-Mistral-7B"

tokenizer = tokenizer_mistral = AutoTokenizer.from_pretrained(model_id)

mistral_token_lens = {k: len(k) for k in tokenizer_mistral.vocab.keys()}

{k: v for k, v in mistral_token_lens.items() if v >16}
# %%
cnt = Counter(mistral_token_lens.values())
sorted(list(cnt.items()), key=lambda x: x[0])

# %%
tokenizer_mistral.vocab
# %%
from transformers import AutoTokenizer, AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained(model_id)
# %%

# Here our goal is to build a encoder that takes in the token's utf-8 bytes and outputs the token's embedding

def str_to_bytetokens(s):
    return list(s.encode('utf-8'))
vocab_1 = next(iter(tokenizer_mistral.vocab.keys()))
vocab_1_bytes = str_to_bytetokens(vocab_1)
vocab_1_bytes

import pandas as pd

df = pd.DataFrame(tokenizer_mistral.vocab.items(), columns=['token', 'id'])
# bytetokens is a list of integers
df['bytetokens'] = df['token'].apply(str_to_bytetokens)
# %%
df['bytetokens'].apply(len).value_counts().sort_index()
# %%
len(tokenizer_mistral.vocab)
# 32000
# %%
model.model.embed_tokens.weight
# Embedding(32002, 4096)
# %%
tokenizer.special_tokens_map
# {'bos_token': '<s>', 'eos_token': '</s>', 'unk_token': '<unk>'}
# %%

# for the sake of simplicity
    # we will adapt only non-special tokens
    # we will not do auto-regressive modeling, but will causal mask
    # we will not use the positional embeddings (NoPE)
# we'll use nanoGPT as the model
# input:
# [<startoftoken>] + bytetokens +  [<endoftoken>]
# left padding, use the rightmost token to pool the token's embedding
# apply mlp layer to the pooled embedding, to upscale to the main model's embedding size

# now let's prepare the data
# %%
def concat_list(ll):
    return [item for l in ll for item in l]
len(Counter(concat_list(df['bytetokens'])))
# %%
max(concat_list(df['bytetokens']))
# %%
# decode int to utf-8
def int_list_to_str(l):
    return bytes(l).decode('utf-8')

int_list_to_str([60, 115, 62])
int_list_to_str([0])
# %%
df['token_len'] = df['token'].apply(len)
df['bytetoken_len'] = df['bytetokens'].apply(len)
# %%
df['bytetoken_len'].max()
# %%

# again, for simplicity, we'll reserve 0-255 for the byte tokens, and 256, 257, 258 for the special tokens
# even though some bytes never appear
# 256: start of token
# 257: end of token
# 258: padding
# %%
def left_pad_to_len(l, n, pad=0):
    return [pad] * (n - len(l)) + l


ids = df['bytetokens'][0]
padded = left_pad_to_len([256] + ids + [257], 50, 258)

len(ids), len(padded)

# %%
df['inputs'] = df['bytetokens'].apply(lambda x: left_pad_to_len([256] + x + [257], 50, 258))

# %%
# %%
model.model.embed_tokens.weight.shape
# %%
model.model.embed_tokens.weight
# %%
# %%
tokenizer.special_tokens_map
# %%
df = df.sort_values('id').reset_index(drop=True)
# %%
df
# %%
target_ids = df['id'].values
target_ids
# %%
# %%
import numpy as np
import os
input_ids = np.stack([np.array(x) for x in df['inputs']]).astype(np.uint16)
# input_ids.dtype, input_ids.max()
import torch
with torch.no_grad():
    out = model.model.embed_tokens(torch.tensor(target_ids))
output_embeddings = out.numpy().astype(np.float32)
# %%
#df.to_parquet('mistral_vocab_bytetokens.parquet')
# export to bin files
#input_ids.tofile(os.path.join(os.path.dirname(__file__), 'input_ids.bin'))
#output_embeddings.tofile(os.path.join(os.path.dirname(__file__), 'output_embeddings.bin'))

import pickle as pkl
pkl.dump({
    'input_ids': input_ids,
    'output_embeddings': output_embeddings
}, open('mistral_vocab_bytetokens.pkl', 'wb'))
# %%
# load
import pickle as pkl
data = pkl.load(open('mistral_vocab_bytetokens.pkl', 'rb'))
input_ids = data['input_ids']
output_embeddings = data['output_embeddings']
input_ids.shape, output_embeddings.shape, input_ids.dtype, output_embeddings.dtype

# %%
# there seems to be 2 extra tokens in the embedding
df
# %%
model.model.embed_tokens.weight.shape
# %%
