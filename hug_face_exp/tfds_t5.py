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
                            TFT5Model, T5Config, pipeline)

# Tensorflow
import tensorflow as tf
import tensorflow_datasets as tfds


#PyTorch
import torch

# WandB – Import the wandb library
import wandb

# %load_ext tensorboard
# -

print(tf.__version__)

# ### Setup Directories

data_dir = "/home/karthikrbabu/data_speaks_e2e/tf_data/model_data"
log_dir = f"{data_dir}/experiments/t5/logs"
save_path = f"{data_dir}/experiments/t5/models"
cache_path_train = f"{data_dir}/cache/t5.train"
cache_path_test = f"{data_dir}/cache/t5.test"

train_df = pd.read_csv('../data/e2e-cleaning-master/cleaned-data/train-fixed.no-ol.csv').drop(["fixed","orig_mr"], axis=1)
dev_df = pd.read_csv('../data/e2e-cleaning-master/cleaned-data/devel-fixed.no-ol.csv').drop(["fixed","orig_mr"], axis=1) 
test_df = pd.read_csv('../data/e2e-cleaning-master/cleaned-data/test-fixed.csv').drop(["fixed","orig_mr"], axis=1)
print("Train Size", train_df.shape)
print("Dev Size", dev_df.shape)
print("Test Size", test_df.shape)
train_df.head()

print(train_df['mr'][0])
print()
print(train_df['ref'][0])

# ### Load Data 

train, info = tfds.load('e2e_cleaned', split='train', with_info=True)
validation = tfds.load('e2e_cleaned', split='validation', with_info=False)
print(info)


# ### Configurable Params

# +
EPOCHS = 6
BATCH_SIZE = 6
WARMUP_STEPS = 1e4
BUFFER_SIZE = 1000
NTRAIN = info.splits["train"].num_examples
NDEV = info.splits["validation"].num_examples
STEPS = int(np.ceil(NTRAIN/BATCH_SIZE))
DEV_STEPS = int(np.ceil(NDEV/BATCH_SIZE))
print("Total Steps: ", STEPS)
print("Total Dev Steps: ", DEV_STEPS)



#Configs
# model_config = {
#     'vocab_size':32100,
#     'd_model':512,
#     'd_kv':64,
#     'd_ff':2048,
#     'num_layers': 6,
#     'num_decoder_layers':None,
#     'num_heads':8,
#     'relative_attention_num_buckets':32,
#     'dropout_rate':0.1,
#     'layer_norm_epsilon':1e-06,
#     'initializer_factor':1.0,
#     'feed_forward_proj':'relu',
#     'is_encoder_decoder':True,
#     'use_cache':True,
#     'pad_token_id': 0,
#     'eos_token_id': 1
# }


token_config = {
    'add_special_tokens': True,
    'padding': 'longest',
    'truncation':True,    
    'max_length':60, #60 chosen after understanding the data flow with the tokenizer
    'stride': 0,
    'return_tensors': 'tf',
    'return_token_type_ids': False,
    'return_attention_mask': True,
    'return_overflowing_tokens': False,
    'return_special_tokens_mask': False,
    'return_offsets_mapping': False,
    'return_length': True,
}
# t5_config = T5Config(**token_config)


config = {
    'model_size': "t5-small",
    'max_len': token_config['max_length']
}


# -

class T5Wrapper(TFT5ForConditionalGeneration):
    def __init__(self, *args, log_dir=None, cache_dir= None, **kwargs):
        super().__init__(*args, **kwargs)
#         self.loss_tracker= tf.keras.metrics.CategoricalAccuracy(name='categorical_accuracy', dtype=None)
        self.loss_tracker= tf.keras.metrics.Mean(name='loss')         
    
    @tf.function
    def train_step(self, data):
        x, _= data
        y = x["labels"]
        y = tf.reshape(y, [-1, 1])
        with tf.GradientTape() as tape:
            outputs = self(x, training=True)
            loss = outputs[0]
            logits = outputs[1]
            loss = tf.reduce_mean(loss)
            
            grads = tape.gradient(loss, self.trainable_variables)
            
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        lr = self.optimizer._decayed_lr(tf.float32)
        
        self.loss_tracker.update_state(loss)        
        self.compiled_metrics.update_state(y, logits)
        metrics = {m.name: m.result() for m in self.metrics}
        metrics.update({'lr': lr})
        
        return metrics

    def test_step(self, data):
        x, _ = data
        y = x["labels"]
        y = tf.reshape(y, [-1, 1])
        output = self(x, training=False)
        loss = output[0]
        loss = tf.reduce_mean(loss)
        logits = output[1]
        
        self.loss_tracker.update_state(loss)
        self.compiled_metrics.update_state(y, logits)
        return {m.name: m.result() for m in self.metrics}


# +
def encode(tags, contents, target_text):
    """
    Encode function that uses the T5 Tokenizer on each example
    """
    print("encode start")

    mr = "data_to_text: "    
    mr_entry_length = len(tags)
    tag_names = tags.numpy()
    content_names = contents.numpy()
    
    for i in range(mr_entry_length):
        tag = tag_names[i].decode('UTF-8')
        content = content_names[i].decode('UTF-8')
        if i == mr_entry_length - 1:
            mr += f"{tag}[{content}] </s>"
        else:
            mr += f"{tag}[{content}], "
    print(f"mr created: {mr}")
    
    target = f"{str(target_text.numpy().decode('utf-8'))}"
    target = f"{target} </s>"
    print(f"target created: {target}")
    
    encoder_inputs = tokenizer(mr,
                               truncation=True, 
                               return_tensors='tf',
                               max_length=token_config['max_length'],
                               pad_to_max_length=True)    
    
    decoder_inputs = tokenizer(target,
                               truncation=True, 
                               return_tensors='tf',
                               max_length=token_config['max_length'],
                               pad_to_max_length=True)  
    print(f"tokenized")
    input_ids = encoder_inputs['input_ids'][0]
    input_attention = encoder_inputs['attention_mask'][0]
    target_ids = decoder_inputs['input_ids'][0]
    target_attention = decoder_inputs['attention_mask'][0]
    
    print(f"encode return")
    return input_ids,input_attention, target_ids, target_attention



def encode_tf(inputs):
    """
    Encode_Tf function that applies our custom encode function on each tensor example
    """
    
    target_text = inputs['target_text']
    print('encode_tf got targert')
    tags = inputs['input_text']['table']['column_header']    
    print('encode_tf got tags')    
    contents = inputs['input_text']['table']['content']
    print('encode_tf got content')    
    
    print('encode_tf starting map')
    encoded = tf.py_function(encode, [tags, contents, target_text],
                                           [tf.int32, tf.int32, tf.int32, tf.int32])
    
    input_ids, input_attention, target_ids, target_attention = encoded
    
    print()
    print("back in encode_tf")
    input_ids.set_shape([None])
    target_ids.set_shape([None])
    input_attention.set_shape([None])
    target_attention.set_shape([None])
    
    print("shapes set")
#     labels = tf.reshape(target_ids, [-1, 1])
    data= {'input_ids': input_ids, #'decoder_input_ids': target_ids, 
            'labels': target_ids, 
            'attention_mask': input_attention,
           'decoder_attention_mask': target_attention}
    return (data, None)




def create_dataset(source_dataset, cache_path=None, batch_size=4, 
                   buffer_size= 1000, shuffling=True):
    """
    Builds data object ready for use given our training dataset in the form of tensors
    """

    dataset = source_dataset.map(encode_tf, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    
    if cache_path is not None:
        dataset = dataset.cache(cache_path)        
    if shuffling:
        dataset = dataset.shuffle(buffer_size)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    return dataset


# -

# ### Init Tokenizer

tokenizer = AutoTokenizer.from_pretrained(config['model_size'])

# ### Process Train/ Validation

train_ds= create_dataset(train, batch_size=batch_size, 
                         shuffling=True, cache_path = None)
print('~~~~~~')
valid_ds = create_dataset(validation, batch_size=batch_size, 
                         shuffling=False, cache_path = None)


# ### Check Data

data = next(iter(train_ds))
print("Example data from the dataset: \n", data)


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
metrics = ['sparse_categorical_accuracy','categorical_accuracy', 'categorical_crossentropy']
# -

# ### Optimizer

learning_rate = CustomSchedule(warmup_steps)
# learning_rate = 0.001  # Instead set a static learning rate
optimizer = tf.keras.optimizers.Adam(learning_rate)

# ### Setup Model

model = T5Wrapper.from_pretrained("t5-base")

# ### Compile Model

model.compile(optimizer=optimizer, metrics=metrics)
model.summary()

# ### Start Tensorboard

# %tensorboard --logdir /home/karthikrbabu/data_speaks_e2e/tf_data/model_data/experiments/t5/logs

epochs_done = 0
model.fit(train_ds, epochs=5, steps_per_epoch=steps, callbacks=callbacks, 
          validation_data=valid_ds, validation_steps=valid_steps, initial_epoch=epochs_done)

# ### Train Model

t5_gen_model.save_pretrained(save_path)



model = TFT5ForConditionalGeneration.from_pretrained('t5-small')
# inputs = train_ds.x['input_ids'][0]
# labels = train_ds.y['input_ids'][0]
# outputs = model(inputs, labels=labels)
# loss = outputs.loss
# logits = outputs.logits
# inputs = tokenizer("summarize: studies have shown that owning a dog is good for you ", return_tensors="tf").input_ids  # Batch size 1
# result = model.generate(inputs)

# +
text = f"""{train['mr'][0]}""".replace('\n', ' ')

encoding = tokenizer.encode("""data to text: """ + text, return_tensors='tf')
outputs = model.generate(encoding,
                      num_beams=4, temperature=6,
                                    no_repeat_ngram_size=2,
                                    min_length=30,
                                    max_length=100,
                                    early_stopping=True)

summarization = tokenizer.decode(outputs[0])
summarization
# -


