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
# Goal of this notebook is to cleanly setup a barebone T5 Model Pipeline to generate text summaries form our meaning representations. Should include the following steps
#
# * Import Libraries
# * Load Train/Dev/Test Data
# * Config Definitions
# * Pre-process Data (Tensors)
# * Quick Tensor EDA
# * Tensorboard Loading
# * Optimizer Init
# * Train Model
# * Evaluate Model

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
                            TFTrainer, TFTrainingArguments, T5Tokenizer, TFT5ForConditionalGeneration,
                            TFT5Model, TFT5EncoderModel, T5Config, pipeline)

# Tensorflow
import tensorflow as tf
# tf.enable_eager_execution()

#PyTorch
import torch

# WandB – Import the wandb library
import wandb

# %load_ext tensorboard
# -

print(tf.__version__)

# ### Setup Directories

data_dir = "./data"
log_dir = f"{data_dir}/experiments/t5/logs"
save_path = f"{data_dir}/experiments/t5/models"
cache_path_train = f"{data_dir}/cache/t5.train"
cache_path_test = f"{data_dir}/cache/t5.test"

# ### Load Data 

train = pd.read_csv('../data/e2e-cleaning-master/cleaned-data/train-fixed.no-ol.csv').drop(["fixed","orig_mr"], axis=1)
dev = pd.read_csv('../data/e2e-cleaning-master/cleaned-data/devel-fixed.no-ol.csv').drop(["fixed","orig_mr"], axis=1) 
test = pd.read_csv('../data/e2e-cleaning-master/cleaned-data/test-fixed.csv').drop(["fixed","orig_mr"], axis=1)
print("Train Size", train.shape)
print("Dev Size", dev.shape)
print("Test Size", test.shape)
train.head()

# ### Configs

# +
#Configs
config = {
    'model_size': "t5-small",
    'max_len': 100
}

model_config = {
    'vocab_size':32100,
    'd_model':512,
    'd_kv':64,
    'd_ff':2048,
    'num_layers': 6,
    'num_decoder_layers':None,
    'num_heads':8,
    'relative_attention_num_buckets':32,
    'dropout_rate':0.1,
    'layer_norm_epsilon':1e-06,
    'initializer_factor':1.0,
    'feed_forward_proj':'relu',
    'is_encoder_decoder':True,
    'use_cache':True,
    'pad_token_id': 0,
    'eos_token_id': 1
}


token_config = {
    'add_special_tokens': True,
    'padding': 'longest',
    'truncation':True,    
    'max_length':None,
    'stride': 0,
    'is_split_into_words':False,
    'return_tensors': 'tf',
    'return_token_type_ids': False,
    'return_attention_mask': True,
    'return_overflowing_tokens': False,
    'return_special_tokens_mask': False,
    'return_offsets_mapping': False,
    'return_length': True,
}
# -

# ### Initializations

# +
tokenizer = AutoTokenizer.from_pretrained(config['model_size'])
# t5_config = T5Config(**token_config)

#Vocab Length
print("Vocab Length: ", len(tokenizer.vocab))


# -

class DataWrapper():
    def __init__(self, df, tokenizer, token_config):
        self.df = df
        self.shape = df.shape
        self.tokenizer = tokenizer
        self.token_config = token_config
        self.x = self.encode(df['mr'])
        self.y = self.encode(df['ref'])


    def encode(self, input_list):
        """
        Pre-process all data thats passed through BatchEncoding
        * input_list - list of strings
        * tokenizer - T5 or Autotokenizer
        * token_config - Config to define encoding options
        """

        #Add 'summary' task prefix to the input
        inputs = ['summarize: ' + s for s in input_list]

        #Process data for training 
        batch_encoding= self.tokenizer(inputs,**self.token_config)

        input_ids = batch_encoding['input_ids']
        attention_mask = batch_encoding['attention_mask']
        lengths = batch_encoding['length']
    
        return {'input_ids':input_ids, 'attention_mask': attention_mask, 'lengths':lengths}


# ### Process Train

#Pre Process Training Data
train_ds = DataWrapper(train, tokenizer, token_config)
print("Input Padded Length: ", train_ds.x['lengths'][0])
print("Output Padded Length: ", train_ds.y['lengths'][0])

train_mrs_token_counts = pd.DataFrame({'counts': [np.count_nonzero(mask) for mask in train_ds.x['attention_mask']]})
print(stats.describe(train_mrs_token_counts['counts']))
fig = px.histogram(train_mrs_token_counts, x="counts", histnorm='percent') # histnorm: percent, probability, density
fig.update_layout(title_text="Histogram: Train Mrs Token Counts")
fig.show()

train_labels_token_counts = pd.DataFrame({'counts': [np.count_nonzero(label) for label in train_ds.y['attention_mask']]})
print(stats.describe(train_labels_token_counts['counts']))
fig = px.histogram(train_labels_token_counts, x="counts", histnorm='percent') # histnorm: percent, probability, density
fig.update_layout(title_text="Histogram: Train Labels Token Counts")
fig.show()


# ### Custom Learning Rate Optimizer

# +
class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    """
    Class created to optimize the learning rate moderation to start steep, then become gradual
    """
    def __init__(self, warmup_steps=1e4):
        super().__init__()

        self.warmup_steps = tf.cast(warmup_steps, tf.float32)
    
    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        m = tf.maximum(self.warmup_steps, step)
        m = tf.cast(m, tf.float32)
        lr = tf.math.rsqrt(m)
    
        return lr 

#Example
plt.style.use('ggplot')
schedule = CustomSchedule()
plt.plot(schedule(tf.range(25000, dtype=tf.float32)))
plt.xlabel("Steps")
plt.ylabel("Learning rate")
# -

# ### Train Parameters
#

warmup_steps = 1e4
batch_size = 4
encoder_max_len = 250
decoder_max_len = 54
buffer_size = 1000
ntrain = train.shape[0]
nvalid = dev.shape[0]
steps = int(np.ceil(ntrain/batch_size))
valid_steps = int(np.ceil(nvalid/batch_size))
print("Total Steps: ", steps)
print("Total Validation Steps: ", valid_steps)

# ### Setup Logging - TensorBoard

# +
start_profile_batch = steps+10
stop_profile_batch = start_profile_batch + 100
profile_range = f"{start_profile_batch},{stop_profile_batch}"

log_path = log_dir + "/" + datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_path, histogram_freq=1,
                                                     update_freq=20,profile_batch=profile_range)

checkpoint_filepath = save_path + "/" + "T5-{epoch:04d}-{val_loss:.4f}.ckpt"
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=False,
    monitor='val_loss',
    mode='min',
    save_best_only=True)

callbacks = [tensorboard_callback, model_checkpoint_callback] 
metrics = [tf.keras.metrics.SparseTopKCategoricalAccuracy(name='accuracy') ]
# -

# ### Optimizer

learning_rate = CustomSchedule(warmup_steps)
# learning_rate = 0.001  # Instead set a static learning rate
optimizer = tf.keras.optimizers.Adam(learning_rate)

# ### Process Dev

#Pre Process Dev Data
dev_ds = DataWrapper(dev, tokenizer, token_config)
print("Input Padded Length: ", dev_ds.x['lengths'][0])
print("Output Padded Length: ", dev_ds.y['lengths'][0])


# ### TensorBoard

# %tensorboard --logdir ./data/experiments/t5/logs

# ### Train Model

# +
def t5_keras_model():
    """
    Use this function to define our basic model pipeline. 
    Following the keras pattern so that we can apply our model as a function on the next layer etc.
    """
    
    #Here we have input and label token_ids, as well as their attention_masks
    encode_in = tf.keras.layers.Input(shape=(config['max_len'],), dtype='int32', name="encode_in_ids")
    enc_mask_in = tf.keras.layers.Input(shape=(config['max_len'],), dtype='int32', name="enc_mask_in_ids")
    decode_in = tf.keras.layers.Input(shape=(None,), dtype='int32', name="decode_in_ids")
    dec_mask_in = tf.keras.layers.Input(shape=(None,), dtype='int32', name="dec_mask_in_ids")
    
    #Pull in our model
    t5_layer = TFT5ForConditionalGeneration.from_pretrained(config['model_size'])
    
    #Pass along all the parameters to the model
    t5_out = t5_layer({'input_ids': encode_in, 
                       'decoder_input_ids':decode_in, 
                       'attention_mask':enc_mask_in,
                       'decoder_attention_mask':dec_mask_in
                      }, return_dict=True)
    
    #Checking output keys 
    print(t5_out.keys())
    
    #Predicted logits
    pred_logits = t5_out['logits']
    
    #Model pipeline
    model = tf.keras.models.Model(inputs=[encode_in, 
                                          enc_mask_in, 
                                          decode_in,
                                          dec_mask_in
                                         ], 
                                  outputs=pred_logits)
    #Compile model
    model.compile(
        optimizer=optimizer, 
        metrics=metrics,
    )

# Option 2 
#     model.compile(
#         optimizer=tf.keras.optimizers.Adam(),
#         metrics=[tf.keras.metrics.Accuracy()]
#     )

    #Print summary
    model.summary()
    return model


# +
tf.keras.backend.clear_session()
try:
    del t5_gen_model
except:
    pass

t5_gen_model = t5_keras_model(train_ds)
# -



















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


