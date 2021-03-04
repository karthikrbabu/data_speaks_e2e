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
    TFTrainer, TFTrainingArguments, T5Tokenizer, TFT5ForConditionalGeneration, TFT5Model, TFT5EncoderModel,T5Config)

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
    'model_size': "t5-small"
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
    'max_length':None,
    'max_target_length':None,
    'padding':'longest',
    'return_tensors': 'tf',
    'truncation':True,
}
# -

# ### Initializations

# +
tokenizer = AutoTokenizer.from_pretrained(config['model_size'])
t5_config = T5Config(**token_config)
t5_layer = TFT5ForConditionalGeneration.from_pretrained(config['model_size'])

#Vocab Length
print("Vocab Length: ", len(tokenizer.vocab))


# +
def convert_pd_to_tf_dataset(data):
    """
    Function responsible for taking our Dataset and converting it to Tensorflow Dataset Object
    https://medium.com/when-i-work-data/converting-a-pandas-dataframe-into-a-tensorflow-dataset-752f3783c168
    """
    
    features = ['mr']
    converted_dataset = (
        tf.data.Dataset.from_tensor_slices(
            (
                tf.cast(data[features].values, tf.float32),
                tf.cast(data['ref'].values, tf.int32)
            )
        )
    )
    
    return converted_dataset


def encode(data, tokenizer, token_config):
    """
    Pre-process all data thats passed through BatchEncoding
    * data - 2 columns ['mr','ref']
    * tokenizer - T5 or Autotokenizer
    * token_config - Config to define encoding options
    """
    df = data.copy()
    
    #Add 'summary' task prefix to the input
    df.mr = 'summarize: ' + df.mr

    #Process data for training 
    batch_encoding= tokenizer.prepare_seq2seq_batch(src_texts=list(df['mr']), 
                                                 tgt_texts=list(df['ref']),
                                                 **token_config)
    input_ids = batch_encoding['input_ids']
    input_attention_mask = batch_encoding['attention_mask']
    label_ids = batch_encoding['labels']
    
    return {'input_ids':input_ids, 'attention_mask': input_attention_mask, 'labels':label_ids}


#[WIP] - might not be needed
def encode_tf(data, tokenizer, token_config):
    encoded = encode(data, tokenizer, token_config)
    return (encoded, None)


#[WIP] - might not be needed
def create_dataset(source_dataset, tokenizer, token_config, 
                   cache_path=None, batch_size=4, 
                   buffer_size= 1000, shuffling=True):

    dataset = encode_tf(source_dataset, tokenizer, token_config)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    return dataset
    


# -

# ### Process Train

#Pre Process Training Data
train_preprop = encode(train, tokenizer, token_config)
train_input_ids, train_attention_mask, train_labels = train_preprop['input_ids'], train_preprop['attention_mask'], train_preprop['labels']
print("Input Padded Length: ", len(train_input_ids[0]))
print("Output Padded Length: ", len(train_labels[0]))

train_mrs_token_counts = pd.DataFrame({'counts': [np.count_nonzero(mask) for mask in train_attention_mask]})
print(stats.describe(train_mrs_token_counts['counts']))
fig = px.histogram(train_mrs_token_counts, x="counts", histnorm='percent') # histnorm: percent, probability, density
fig.update_layout(title_text="Histogram: Train Mrs Token Counts")
fig.show()

train_labels_token_counts = pd.DataFrame({'counts': [np.count_nonzero(label) for label in train_labels]})
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

# ### Optimizer

learning_rate = CustomSchedule(warmup_steps)
# learning_rate = 0.001  # Instead set a static learning rate
optimizer = tf.keras.optimizers.Adam(learning_rate)

# ### Process Dev

#Pre Process Training Data
dev_preprop = encode(dev, tokenizer, token_config)
dev_input_ids, dev_attention_mask, dev_labels = dev_preprop['input_ids'], dev_preprop['attention_mask'], dev_preprop['labels']
print("Input Padded Length: ", len(dev_input_ids[0]))
print("Output Padded Length: ", len(dev_labels[0]))

# ### TensorBoard

# %tensorboard --logdir ./data/experiments/t5/logs

# ### Train Model

t5_layer.compile(optimizer=optimizer, metrics=metrics)

epochs_done = 0
t5_layer.fit(train_ds, epochs=5, steps_per_epoch=steps, callbacks=callbacks, 
          validation_data=valid_ds, validation_steps=valid_steps, initial_epoch=epochs_done)



















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





