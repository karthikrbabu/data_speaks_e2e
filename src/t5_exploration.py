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

# # T5 Model Exploration
#
# Take a look at the initial spread of inputs once we tokenize our data.

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
import transformers
from transformers import (TFAutoModelWithLMHead, AutoTokenizer, 
                            TFTrainer, TFTrainingArguments, T5Tokenizer, TFT5ForConditionalGeneration,
                            TFT5Model, T5Config, pipeline)


import datasets
from datasets import load_dataset, list_datasets

# Tensorflow
import tensorflow as tf
import tensorflow_datasets as tfds


# +
tf_version = tf.__version__
print("Tensorflow: ", tf_version)
print("Transformers: ", transformers.__version__)
print("Datasets: ", datasets.__version__)

tf_version_split = tf_version.split('.')
assert int(tf_version_split[0])==2 and int(tf_version_split[-2])>=3, f"Tensorflow version should be '2.3+,x', given {tf_version}"

# -

# ### Load Data 

train = pd.read_csv('../data/e2e-cleaning-master/cleaned-data/train-fixed.no-ol.csv').drop(["fixed","orig_mr"], axis=1)
dev = pd.read_csv('../data/e2e-cleaning-master/cleaned-data/devel-fixed.no-ol.csv').drop(["fixed","orig_mr"], axis=1) 
test = pd.read_csv('../data/e2e-cleaning-master/cleaned-data/test-fixed.csv').drop(["fixed","orig_mr"], axis=1)
print("Train Size", train.shape)
print("Dev Size", dev.shape)
print("Test Size", test.shape)
train.head()

# ### Initializations

# +
tokenizer = AutoTokenizer.from_pretrained('t5-small')
encoder_max_len = 60
decoder_max_len = 60

#Vocab Length
print("Vocab Length: ", len(tokenizer))


# +
def encode_mr_to_len(mr):
    """
    Return length after tokenization for MR 
    """
    mr_base = f"data_to_text: {str(mr)} </s>"

    encoder_inputs = tokenizer(mr_base, truncation=True, return_tensors='tf', pad_to_max_length=True)
    
    input_ids = encoder_inputs['input_ids'][0]
    input_attention = encoder_inputs['attention_mask'][0]
    return sum(input_attention.numpy())


def encode_ref_to_len(ref):
    """
    Return length after tokenization for REF
    """
    ref_base = f"{str(ref)} </s>"
    decoder_inputs = tokenizer(ref_base, truncation=True, return_tensors='tf', pad_to_max_length=True)
    
    target_ids = decoder_inputs['input_ids'][0]
    target_attention = decoder_inputs['attention_mask'][0]
    return sum(target_attention.numpy())



# -

train['mr_token_length'] = train['mr'].apply(encode_mr_to_len)
train['ref_token_length'] = train['ref'].apply(encode_mr_to_len)

train_mrs_token_counts = pd.DataFrame({'counts': train['mr_token_length']})
print(stats.describe(train_mrs_token_counts['counts']))
fig = px.histogram(train_mrs_token_counts, x="counts", histnorm='percent') # histnorm: percent, probability, density
fig.update_layout(title_text="Histogram: Train Mrs Token Counts")
fig.show()

train_mrs_token_counts = pd.DataFrame({'counts': train['mr_token_length']})
print(stats.describe(train_mrs_token_counts['counts']))
fig = px.histogram(train_mrs_token_counts, x="counts", histnorm='percent', cumulative=True) # histnorm: percent, probability, density
fig.update_layout(title_text="Cumulative Histogram: Train Mrs Token Counts")
fig.show()

# #### 99% is captured by 60 tokens

train_labels_token_counts = pd.DataFrame({'counts': train['ref_token_length']})
print(stats.describe(train_labels_token_counts['counts']))
fig = px.histogram(train_labels_token_counts, x="counts", histnorm='percent') # histnorm: percent, probability, density
fig.update_layout(title_text="Histogram: Train Labels Token Counts")
fig.show()

train_labels_token_counts = pd.DataFrame({'counts': train['ref_token_length']})
print(stats.describe(train_labels_token_counts['counts']))
fig = px.histogram(train_labels_token_counts, x="counts", histnorm='percent', cumulative=True) # histnorm: percent, probability, density
fig.update_layout(title_text="Histogram: Train Labels Token Counts")
fig.show()

# #### 99% is captured by 50 tokens



