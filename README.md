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
- run evaluation
    - it's done in evaluate.py (%-style notebook)
- test the trained embedding
    - `python test_mistral.py`

### V0.1 Result

- ~~`The trainging log shows the loss converges to 0 in 0.0000 after 56 iters. Takes a few seconds on a 3090 GPU.~~
    - update: after visual inspection, the model learned to predict the same embedding for all inputs. Changing the hparams and allow it to train longer fixed it.
- Training it for 10k iters took ~20mins on a 3090 GPU.
- Evaluation:
    - Embedding accuracy: I used the trained and original embedding to calculate softmax. The resulting accuracy is 0.9883.
    Meaning the model learned to differentiate between tokens correctly. (see evaluate.py for details)
    - End-to-end test: Replace the token embedding in the main model with the trained embedding. The model can still generate text, but it sometimes generates typo and eventually diverges. (see test_mistral.py for details)

- TODO:
    - [x] verify the training is working as expected
    - [x] plug this model to mistral and see how it performs
    - [ ] plug this model to mistral and further train end-to-end
    - [ ] Try to combine softmax loss (currently only used L1 and MSE loss)


## Future Directions / Discussions / Ideas


- This doesn't replace the LM head in the main model, but I think we can use loss to pull LM head closer to the token embedding (or vice versa) to make the LM head also have char level understanding.

- If we want to adapt the idea from [MEGABYTE](https://arxiv.org/abs/2305.07185) to also replace the LM head with subtoken-lavel decoder, we need to make sure the encoder can handle possible out-of-vocabulary tokens during inference.

- We can potentially train a NN based tokenizer with a VAE-like structure. And token baundry can be decided by the model (maybe based on reconstruction loss).


# LICENSE
The model and training file is copied and adapted from https://github.com/karpathy/nanoGPT/. MIT LICENSE



