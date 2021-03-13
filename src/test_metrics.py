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
import sys
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

# -

#Custom Utils Lib
from utils.utils import get_model_output, write_out_tsv, encode, to_tf_dataset, create_dataset
from classes.t5Wrapper import T5Wrapper
from classes.customScheduler import CustomSchedule

# ### Load a prior model

# !ls /home/ubuntu/karthik/data_speaks_e2e/model_runs

ts_val= '20210311_1024'
model_path = f'/home/ubuntu/karthik/data_speaks_e2e/model_runs/ts={ts_val}'

loaded_model = T5Wrapper.from_pretrained(f'{model_path}')
tokenizer = AutoTokenizer.from_pretrained('t5-small')

# ### Load in Dataset

# +
train = load_dataset('e2e_nlg_cleaned', split='train')
validation = load_dataset('e2e_nlg_cleaned', split='validation')

train.features
# -

train_ds = train.map(lambda x: encode(x, tokenizer))
valid_ds = validation.map(lambda x: encode(x, tokenizer))

tf_train_ds = to_tf_dataset(train_ds)
tf_valid_ds = to_tf_dataset(valid_ds)

tf_train_ds= create_dataset(tf_train_ds, batch_size=30,
                         shuffling=True, cache_path = None)
tf_valid_ds = create_dataset(tf_valid_ds, batch_size=30, 
                         shuffling=False, cache_path = None)

# ### Generate Results

# +
gen_params = {'num_beams': 1, 
              'max_length': 60,
              'min_length': 20, 
              'early_stopping': True,
              'do_sample': False, 
              'no_repeat_ngram_size': 2 
             }

model_ouput = get_model_output(loaded_model, tokenizer, {}, None, tf_valid_ds, None)

# -
# ### Write Output

v_out = model_ouput['validation_output']
write_out_tsv(valid_ds, "validation", v_out, write_path=model_path)


# <hr>

# ### Let's Use E2E Evaluation Metrics

# ls

# cd ..


