# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
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

# The purpose of this notebook is to demonstrate training using tensorflow 2 and keras. This notebook includes tf Data pipelines for build any other NLP task in a text to text fashion. Anyone can adapt the data pipeline to thier own datasets. Uses the efficient [Datasets](https://github.com/huggingface/datasets) from 🤗 as source for training.
# #### Features
# - Train TF T5 on E2E Cleaned Data to Text Problem
# - Train T5 using keras trainer fucntion
# - tf.Data pipeline
# - [Datasets from 🤗](https://github.com/huggingface/datasets) as source
# - Log metrics using tensorboard
# - Profile your experiment with the brand new tensorflow profiler !!

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

# ### Setup Directories

data_dir = "/home/karthikrbabu/data_speaks_e2e/tf_data/model_data"
log_dir = f"{data_dir}/experiments/t5/logs"
save_path = f"{data_dir}/experiments/t5/models"
cache_path_train = f"{data_dir}/cache/t5.train"
cache_path_test = f"{data_dir}/cache/t5.test"


# ### Defining the Model

class T5Wrapper(TFT5ForConditionalGeneration):
    def __init__(self, *args, log_dir=None, cache_dir= None, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_tracker= tf.keras.metrics.Mean(name='loss') 
    
    @tf.function
    def train_step(self, data):
        x = data
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
        x = data
        y = x["labels"]
        y = tf.reshape(y, [-1, 1])
        output = self(x, training=False)
        loss = output[0]
        loss = tf.reduce_mean(loss)
        logits = output[1]
        
        self.loss_tracker.update_state(loss)
        self.compiled_metrics.update_state(y, logits)
        return {m.name: m.result() for m in self.metrics}
        


# ### Init Tokenizer

tokenizer = AutoTokenizer.from_pretrained('t5-small')

# ### Process Train/ Validation

# +
train = load_dataset('e2e_nlg_cleaned', split='train')
validation = load_dataset('e2e_nlg_cleaned', split='validation')

train.features
# -

data = next(iter(train))
print("Example data from the dataset: \n", data)

# ### Init Config

# +
EPOCHS = 6
BATCH_SIZE = 6
WARMUP_STEPS = 1e4
BUFFER_SIZE = 1000
NTRAIN = len(train)
NDEV = len(validation)
STEPS = int(np.ceil(NTRAIN/BATCH_SIZE))
VALID_STEPS = int(np.ceil(NDEV/BATCH_SIZE))
print("Total Steps: ", STEPS)
print("Total Validation Steps: ", VALID_STEPS)


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
# token_config = {
#     'add_special_tokens': True,
#     'padding': 'longest',
#     'truncation':True,    
#     'max_length':60, #60 chosen after understanding the data flow with the tokenizer
#     'stride': 0,
#     'return_tensors': 'tf',
#     'return_token_type_ids': False,
#     'return_attention_mask': True,
#     'return_overflowing_tokens': False,
#     'return_special_tokens_mask': False,
#     'return_offsets_mapping': False,
#     'return_length': True,
# }
# # t5_config = T5Config(**token_config)

config = {
    'model_size': "t5-small",
    'max_len': 60
}


# -

# ### Data Pipeline

def encode(example):
    """
    Encode function that uses the T5 Tokenizer on each example
    """
    
    mr = example['meaning_representation']
    ref = example['human_reference']
  
    mr_base = f"data_to_text: {str(mr)} </s>"
    ref_base = f"{str(ref)} </s>"
    
    encoder_inputs = tokenizer(mr_base,
                               truncation=True,
                               return_tensors='tf',
                               max_length=config['max_len'],
                               pad_to_max_length=True)  
    
    decoder_inputs = tokenizer(ref_base,
                               truncation=True, 
                               return_tensors='tf',
                               max_length=config['max_len'],
                               pad_to_max_length=True)
    
    input_ids = encoder_inputs['input_ids'][0]
    input_attention = encoder_inputs['attention_mask'][0]
    target_ids = decoder_inputs['input_ids'][0]
    target_attention = decoder_inputs['attention_mask'][0]
    
    outputs = {'input_ids':input_ids, 'attention_mask': input_attention, 
               'labels':target_ids, 'decoder_attention_mask':target_attention}
    return outputs
    


# ### Process Train/Validation

train_ds = train.map(encode)
valid_ds = validation.map(encode)

train_ds

ex = next(iter(train_ds))
print("Example data from the mapped dataset: \n", ex)


def to_tf_dataset(dataset):
    """
    Encode_Tf function that applies our custom encode function on each tensor example
    """
    columns = ['input_ids', 'attention_mask', 'labels', 'decoder_attention_mask']
    dataset.set_format(type='tensorflow', columns=columns)
    return_types = {'input_ids':tf.int32, 'attention_mask':tf.int32, 
                'labels':tf.int32, 'decoder_attention_mask':tf.int32}
    return_shapes = {'input_ids': tf.TensorShape([None]), 'attention_mask': tf.TensorShape([None]), 
                  'labels': tf.TensorShape([None]), 'decoder_attention_mask':tf.TensorShape([None])}
    ds = tf.data.Dataset.from_generator(lambda : dataset, return_types, return_shapes)
    return ds
  


# ### Process Train/Validation =>  Tensors

tf_train_ds = to_tf_dataset(train_ds)
tf_valid_ds = to_tf_dataset(valid_ds)

ex = next(iter(tf_train_ds))
print("Example data from the mapped dataset: \n", ex)


def create_dataset(dataset, cache_path=None, batch_size=BATCH_SIZE, 
                   buffer_size= BUFFER_SIZE, shuffling=True):
    """
    Builds data object ready for use given our training dataset in the form of tensors
    """
    
    if cache_path is not None:
        dataset = dataset.cache(cache_path)        
    if shuffling:
        dataset = dataset.shuffle(buffer_size)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    return dataset
    


# ### Build Train/ Validation =>  Model Ready Input

tf_train_ds_final= create_dataset(tf_train_ds, batch_size=BATCH_SIZE, 
                         shuffling=True, cache_path = None)
tf_valid_ds_final = create_dataset(tf_valid_ds, batch_size=BATCH_SIZE, 
                         shuffling=False, cache_path = None)


# ### Custom Learning Rate Scheduler

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

# ### Setup Callbacks for Tensorboard

# +
start_profile_batch = STEPS+10
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

learning_rate = CustomSchedule() # learning_rate = 0.001  # Instead set a static learning rate
optimizer = tf.keras.optimizers.Adam(learning_rate)

# ### Init Model

model = T5Wrapper.from_pretrained(config['model_size'])

model.compile(optimizer=optimizer, metrics=metrics)
model.summary()

# ### Start Tensorboard
#

# %load_ext tensorboard
# %tensorboard --logdir /home/karthikrbabu/data_speaks_e2e/tf_data/model_data/experiments/t5/logs

epochs_done = 0
model.fit(tf_train_ds, epochs=EPOCHS, steps_per_epoch=STEPS, callbacks=callbacks, 
          validation_data=tf_valid_ds, validation_steps=VALID_STEPS, initial_epoch=epochs_done)

iter(tf_train_ds)
# tf_valid_ds

epochs_done = 0
model.fit(iter(tf_train_ds), epochs=EPOCHS, steps_per_epoch=STEPS, callbacks=callbacks, 
          validation_data=tf_valid_ds, validation_steps=VALID_STEPS, initial_epoch=epochs_done)

# +
#upper bound on our data is 33,525
cut_off = 35000

model.fit([train_ds[:cut_off], 
                      train_enc_in_masks_t5[:cut_off], 
                      train_ner_translations_input_t5[:cut_off],
                      train_dec_in_masks_t5[:cut_off]
                     ],
                   train_ner_translations_labels_t5[:cut_off],
                 validation_data=([test_input_sentences_t5[:cut_off],
                                   test_enc_in_masks_t5[:cut_off], 
                                   test_ner_translations_input_t5[:cut_off],
                                   test_dec_in_masks_t5[:cut_off]],
                                test_ner_translations_labels_t5[:cut_off]),
                 batch_size=8,
                epochs=6)
# -


