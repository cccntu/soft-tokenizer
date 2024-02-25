# Soft tokenizer

- This is a work in progress. The goal is to fix some of the issues with LLMs caused by the tokenization process.

## V0

- The first version uses a tiny GPT2 model to train a bytes -> token embedding model.
- Some simpliciations were made to the model, such as removing the positional embedding, removing autoregressive training, and uses a projection at the last position to get the token embedding.

- Notes:
    - mistral tokenizer is nice, because the longest token is only 16 characters, which makes it easier to train a bytes -> token embedding model.
    - after converting to utf-8, the longest token is 48 bytes. So I pad all inputs to 50 sub-token. (including the start and end token)
    - I noticed some special bytes like `\x01` are shows as "<0x00>" in the hf tokenizer. And it's further tokenized into "<", "0", "x", "00", ">", sub-tokens. This is fine for now, but we might need to handle this in the future.
### Getting Started

- prepare the data
    - `python prepare_mistral_data.py`
- train the model
    - `python train.py`

### V0 Result

- The trainging log shows the loss converges to 0 in 0.0000 after 56 iters. Takes a few seconds on a 3090 GPU.
- TODO:
    - [ ] verify the training is working as expected
    - [ ] plug this model to mistral and see how it performs
    - [ ] plug this model to mistral and further train end-to-end


## Future Directions / Discussions / Ideas


- This doesn't replace the LM head in the main model, but I think we can use loss to pull LM head closer to the token embedding (or vice versa) to make the LM head also have char level understanding.

- If we want to adapt the idea from [MEGABYTE](https://arxiv.org/abs/2305.07185) to also replace the LM head with subtoken-lavel decoder, we need to make sure the encoder can handle possible out-of-vocabulary tokens during inference.

- We can potentially train a NN based tokenizer with a VAE-like structure. And token baundry can be decided by the model (maybe based on reconstruction loss).


# LICENSE
The model and training file is copied and adapted from https://github.com/karpathy/nanoGPT/. MIT LICENSE



