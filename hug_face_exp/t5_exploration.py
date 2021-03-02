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
    TFTrainer, TFTrainingArguments, T5Tokenizer, TFT5ForConditionalGeneration, T5Config)

# Tensorflow
import tensorflow as tf

# WandB – Import the wandb library
import wandb

# %load_ext tensorboard
# -

# ### Load Data 

# +
train = pd.read_csv('../data/e2e-cleaning-master/cleaned-data/train-fixed.no-ol.csv')
dev = pd.read_csv('../data/e2e-cleaning-master/cleaned-data/devel-fixed.no-ol.csv') 
test = pd.read_csv('../data/e2e-cleaning-master/cleaned-data/test-fixed.csv') 
print("Train Size", train.shape)
print("Dev Size", dev.shape)
print("Test Size", test.shape)

train.head()
# -

tokenizer = AutoTokenizer.from_pretrained("t5-base")
t5_layer = TFT5ForConditionalGeneration.from_pretrained('t5-base')

encoded = tokenizer.encode("Hello, I'm a single sentence!")


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


