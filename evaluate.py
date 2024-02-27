# %%
import os
import pickle
from contextlib import nullcontext
import torch
import tiktoken
from model import GPTConfig, GPT
import math
import seaborn as sns

# -----------------------------------------------------------------------------
init_from = 'resume' # either 'resume' (from an out_dir) or a gpt2 variant (e.g. 'gpt2-xl')
out_dir = 'out1' # ignored if init_from is not 'resume'
start = "\n" # or "<|endoftext|>" or etc. Can also specify a file, use as: "FILE:prompt.txt"
num_samples = 10 # number of samples to draw
max_new_tokens = 500 # number of tokens generated in each sample
temperature = 0.8 # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
top_k = 200 # retain only the top_k most likely tokens, clamp others to have 0 probability
seed = 1337
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32' or 'bfloat16' or 'float16'
compile = False # use PyTorch 2.0 to compile the model to be faster
#exec(open('configurator.py').read()) # overrides from command line or config file
# -----------------------------------------------------------------------------

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# model
if init_from == 'resume':
    # init from a model saved in a specific directory
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    gptconf = GPTConfig(**checkpoint['model_args'])
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
elif init_from.startswith('gpt2'):
    # init from a given GPT-2 model
    model = GPT.from_pretrained(init_from, dict(dropout=0.0))

model.eval()
model.to(device)
if compile:
    model = torch.compile(model) # requires PyTorch 2.0 (optional)

# look for the meta pickle in case it is available in the dataset folder
# TODO: there are some
# %%
'config' in checkpoint
# %%
checkpoint['config']
# %%
load_meta = True
import pandas as pd
if init_from == 'resume' and 'config' in checkpoint and 'dataset' in checkpoint['config']: # older checkpoints might not have these...
    #meta_path = os.path.join('data', checkpoint['config']['dataset'], 'meta.pkl')
    meta_path = 'meta.pkl'
    load_meta = os.path.exists(meta_path)
if load_meta:
    metadf = pd.read_pickle(meta_path)
    print(f"Loading meta from {meta_path}...")
    """
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    # TODO want to make this more general to arbitrary encoder/decoder schemes
    stoi, itos = meta['stoi'], meta['itos']
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ''.join([itos[i] for i in l])
    """
else:
    # ok let's assume gpt-2 encodings by default
    print("No meta.pkl found, assuming GPT-2 encodings...")
    enc = tiktoken.get_encoding("gpt2")
    encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
    decode = lambda l: enc.decode(l)
# %%

model = model.to(device)
# %%
# let's evaulate the mse of each token in the embedding
import pickle as pkl
data = pkl.load(open('mistral_vocab_bytetokens.pkl', 'rb'))
input_ids = data['input_ids']
output_embeddings = data['output_embeddings']
input_ids.shape, output_embeddings.shape, input_ids.dtype, output_embeddings.dtype
# %%
import numpy as np
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler

unoptimized_model = model
model = torch.compile(model) # requires PyTorch 2.0

batch_size = 640
outputs = []
for i in range(0, len(input_ids), batch_size):
    x = torch.from_numpy(input_ids[i:i+batch_size, :].astype(np.int64))
    x = x.pin_memory().to(device, non_blocking=True)

    with torch.no_grad():
        logits, loss = model(x)
    outputs.append(logits.cpu())

# %%
predicted_embeddings = torch.cat(outputs, dim=0)
original_embeddings = torch.from_numpy(output_embeddings)
# %%
original_norm = original_embeddings.norm(dim=1)
sns.histplot(original_norm.cpu().numpy())
# %%
predicted_norm = predicted_embeddings.norm(dim=1)
sns.histplot(predicted_norm.cpu().numpy())
# %%
df = metadf
# in teknium/OpenHermes-2.5-Mistral-7B
# <|im_end|>and <|im_start|> has a very high norm
# %%
original_embeddings
# %%
sns.heatmap(predicted_embeddings.cpu().numpy()[:100, :100])
# %%
sns.heatmap(original_embeddings.cpu().numpy()[:100, :100])
# %%
sns.heatmap(predicted_embeddings.cpu().numpy()[100:200, :100])
# %%
sns.heatmap(original_embeddings.cpu().numpy()[100:200, :100])
model._orig_mod.final_proj_multiplier
# %%
# check if the closest token is the correct one
# let's calculate the softmax of the embeddings
import torch.nn.functional as F
with torch.no_grad():
    scores = F.softmax(torch.matmul(predicted_embeddings.to('cuda'), original_embeddings.T.to('cuda')), dim=1)
# %%
import gc
gc.collect()
torch.cuda.empty_cache()
# %%
sns.heatmap(scores.cpu().numpy()[-2000:-1900, -2000:-1900])
# %%
with torch.no_grad():
    acc = (scores.argmax(dim=1) == torch.arange(len(scores), device=scores.device)).float().mean()
acc
# 0.9883. not bad, for a model that's not directly trained on this objective
# %%
norm_scale_fix = (original_embeddings).norm(dim=1).mean()/predicted_embeddings.norm(dim=1).mean()
print(f'{norm_scale_fix=}')
new_embeddings = predicted_embeddings * norm_scale_fix

# save the new embeddings, we'll test them in the next notebook
torch.save(new_embeddings, 'new_embeddings.pt')
