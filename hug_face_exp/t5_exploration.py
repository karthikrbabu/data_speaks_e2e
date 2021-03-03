# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.10.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# +
#Utilities
import pandas as pd
import numpy as np
from scipy import stats
import math 
import json
import re
import os
import datetime
import time

import collections
from collections import defaultdict

#Plotting
import matplotlib.pyplot as plt
import plotly.express as px

#NLTK 
from nltk.corpus import stopwords
import nltk

#HuggingFace
from transformers import (TFAutoModelWithLMHead, AutoTokenizer, 
    TFTrainer, TFTrainingArguments, T5Tokenizer, TFT5ForConditionalGeneration, TFT5Model, TFT5EncoderModel,T5Config)

# Tensorflow
import tensorflow as tf

#PyTorch
import torch

# WandB – Import the wandb library
import wandb

# %load_ext tensorboard
# -

# ### Load Data 

# +
train = pd.read_csv('../data/e2e-cleaning-master/cleaned-data/train-fixed.no-ol.csv').drop(["fixed","orig_mr"], axis=1)
dev = pd.read_csv('../data/e2e-cleaning-master/cleaned-data/devel-fixed.no-ol.csv').drop(["fixed","orig_mr"], axis=1) 
test = pd.read_csv('../data/e2e-cleaning-master/cleaned-data/test-fixed.csv').drop(["fixed","orig_mr"], axis=1)
print("Train Size", train.shape)
print("Dev Size", dev.shape)
print("Test Size", test.shape)

train.head()

# +
tokenizer = AutoTokenizer.from_pretrained("t5-small")

t5_config = T5Config(vocab_size=32100, 
                     d_model=512,
                     d_kv=64, 
                     d_ff=2048,
                     num_layers=6,
                     num_decoder_layers=None,
                     num_heads=8, 
                     relative_attention_num_buckets=32,
                     dropout_rate=0.1, 
                     layer_norm_epsilon=1e-06, 
                     initializer_factor=1.0, 
                     feed_forward_proj='relu',
                     is_encoder_decoder=True,
                     use_cache=True, 
                     pad_token_id=0, 
                     eos_token_id=1)

t5_layer = TFT5ForConditionalGeneration.from_pretrained('t5-small')


# -

#Vocab Length
len(tokenizer.vocab)

# +
# %%time

#Process data for training 
batch_encoding= tokenizer.prepare_seq2seq_batch(src_texts=list(train['mr']), 
                                             tgt_texts=list(train['ref']),
                                             max_length=None,
                                             max_target_length=None,
                                             padding='longest',
                                             return_tensors='tf',
                                             truncation=True)
batch_encoding.keys()
# -

print("Input Padded Length: ", len(batch_encoding.input_ids[0]))
print("Output Padded Length: ", len(batch_encoding.labels[0]))

train_mrs_token_counts = pd.DataFrame({'counts': [np.count_nonzero(mask) for mask in batch_encoding.attention_mask]})
print(stats.describe(train_mrs_token_counts['counts']))
fig = px.histogram(train_mrs_token_counts, x="counts", histnorm='percent') # histnorm: percent, probability, density
fig.update_layout(title_text="Histogram: Train Mrs Token Counts")
fig.show()

train_labels_token_counts = pd.DataFrame({'counts': [np.count_nonzero(label) for label in batch_encoding.labels]})
print(stats.describe(train_labels_token_counts['counts']))
fig = px.histogram(train_labels_token_counts, x="counts", histnorm='percent') # histnorm: percent, probability, density
fig.update_layout(title_text="Histogram: Train Labels Token Counts")
fig.show()

# ## Example #1

tokenizer = T5Tokenizer.from_pretrained('t5-small')
model = TFT5Model.from_pretrained('t5-small')
input_ids = tokenizer("Studies have been shown that owning a dog is good for you", return_tensors="tf").input_ids  # Batch size 1
decoder_input_ids = tokenizer("Studies show that", return_tensors="tf").input_ids  # Batch size 1
outputs = model(input_ids, decoder_input_ids=decoder_input_ids)


outputs

# ## Example #2

tokenizer = T5Tokenizer.from_pretrained('t5-small')
model = TFT5ForConditionalGeneration.from_pretrained('t5-small')
inputs = tokenizer('The <extra_id_0> walks in <extra_id_1> park', return_tensors='tf').input_ids
labels = tokenizer('<extra_id_0> cute dog <extra_id_1> the <extra_id_2> </s>', return_tensors='tf').input_ids
outputs = model(inputs, labels=labels)
loss = outputs.loss
logits = outputs.logits
inputs = tokenizer("summarize: studies have shown that owning a dog is good for you ", return_tensors="tf").input_ids  # Batch size 1
result = model.generate(inputs)

result.numpy()[0]

print(result)
print(tokenizer.convert_ids_to_tokens(result.numpy()[0]))



# +
# Creating a custom dataset for reading the dataframe and loading it into the dataloader to pass it to the neural network at a later stage for finetuning the model and to prepare it for predictions

class CustomDataset(Dataset):
    
    def __init__(self, dataframe, tokenizer, source_len, summ_len):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.source_len = source_len
        self.summ_len = summ_len
        self.text = self.data.text
        self.ctext = self.data.ctext

    def __len__(self):
        return len(self.text)

    def __getitem__(self, index):
        ctext = str(self.ctext[index])
        ctext = ' '.join(ctext.split())

        text = str(self.text[index])
        text = ' '.join(text.split())

        source = self.tokenizer.batch_encode_plus([ctext], max_length= self.source_len, pad_to_max_length=True,return_tensors='pt')
        target = self.tokenizer.batch_encode_plus([text], max_length= self.summ_len, pad_to_max_length=True,return_tensors='pt')

        source_ids = source['input_ids'].squeeze()
        source_mask = source['attention_mask'].squeeze()
        target_ids = target['input_ids'].squeeze()
        target_mask = target['attention_mask'].squeeze()

        return {
            'source_ids': source_ids.to(dtype=torch.long), 
            'source_mask': source_mask.to(dtype=torch.long), 
            'target_ids': target_ids.to(dtype=torch.long),
            'target_ids_y': target_ids.to(dtype=torch.long)
        }
# -


