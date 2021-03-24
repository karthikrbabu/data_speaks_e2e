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

# # T5 Model Base Implementation
#
# The purpose of this notebook is to demonstrate training using tensorflow 2 and keras. This notebook includes tf Data pipelines for build any other NLP task in a text to text fashion. Anyone can adapt the data pipeline to thier own datasets. Uses the efficient [Datasets](https://github.com/huggingface/datasets) from 🤗 as source for training.
#
# #### Features:
# - Train TF T5 on E2E Cleaned Data to Text Problem
# - Train T5 using keras trainer fucntion
# - tf.Data pipeline
# - [Datasets from 🤗](https://github.com/huggingface/datasets) as source
# - Log metrics using tensorboard
# - Profile your experiment with the brand new tensorflow profiler !!
#
# #### Steps:
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
import warnings
warnings.filterwarnings('ignore')

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

#AWS
import boto3
s3 = boto3.resource('s3')
# -


exp_dir = os.getcwd()
print("Experiment Dir: ", exp_dir)

base_dir = os.path.abspath(os.path.join(os.getcwd(),os.pardir, os.pardir, os.pardir))
os.chdir(base_dir)
print("Base Dir: ", base_dir)

#Custom Utils Lib
from src.utils.utils import (get_model_output, write_model_output, save_metrics,
                         encode, to_tf_dataset, create_dataset, compute_metrics, save_model_to_s3)
from src.classes.t5Wrapper import T5Wrapper
from src.classes.customScheduler import CustomSchedule

# +
tf_version = tf.__version__
print("Tensorflow: ", tf_version)
print("Transformers: ", transformers.__version__)
print("Datasets: ", datasets.__version__)

tf_version_split = tf_version.split('.')
assert int(tf_version_split[0])==2 and int(tf_version_split[-2])>=3, f"Tensorflow version should be '2.3+,x', given {tf_version}"

# -

# ### Setup Directories

# !ls {base_dir}

# +
#AWS box path we should keep
tb_data_dir = f"{exp_dir}/tf_data"
log_dir = f"{tb_data_dir}/experiments/t5/logs"
save_path = f"{tb_data_dir}/experiments/t5/models"
cache_path_train = f"{tb_data_dir}/cache/t5.train"
cache_path_test = f"{tb_data_dir}/cache/t5.test"

print("Experiment Base directory: ",exp_dir)
model_path = f'{exp_dir}/model'
print('model_path: ', model_path)
# -

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
warmup_steps = 1e4
epochs = 5
batch_size = 30
encoder_max_len = 80
decoder_max_len = 80
buffer_size = 1000
ntrain = len(train)
nvalid = len(validation)
steps = int((ntrain//epochs)// batch_size)
valid_steps = int((nvalid//epochs)// batch_size)

print("Train Data Length: ", ntrain)
print("Validation Data Length: ", nvalid)
print("Total Steps: ", steps)
print("Total Validation Steps: ", valid_steps)
print("Batch Size: ", batch_size)
print("Total Epochs: ", epochs)
# -

# ## Data Pipeline

# ### Process Train/Validation

train_ds = train.map(lambda x: encode(x, tokenizer, encoder_max_len, decoder_max_len))
valid_ds = validation.map(lambda x: encode(x, tokenizer, encoder_max_len, decoder_max_len))

ex = next(iter(train_ds))
print("Example data from the mapped dataset: \n", ex)

# ### Process Train/Validation =>  Tensors

tf_train_ds = to_tf_dataset(train_ds)
tf_valid_ds = to_tf_dataset(valid_ds)

# ### Build Train/ Validation =>  Model Ready Input

tf_train_ds= create_dataset(tf_train_ds, batch_size=batch_size, 
                         shuffling=True, cache_path = None)
tf_valid_ds = create_dataset(tf_valid_ds, batch_size=batch_size, 
                         shuffling=False, cache_path = None)

# ### Custom Learning Rate Scheduler

#Example
plt.style.use('ggplot')
schedule = CustomSchedule()
plt.plot(schedule(tf.range(25000, dtype=tf.float32)))
plt.xlabel("Steps")
plt.ylabel("Learning rate")


# ### Setup Callbacks for Tensorboard

save_path

# +
# start_profile_batch = steps+10
# stop_profile_batch = start_profile_batch + 100
# profile_range = f"{start_profile_batch},{stop_profile_batch}"

# log_path = log_dir + "/" + datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
# tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_path, histogram_freq=1,
#                                                      update_freq=20,profile_batch=profile_range)

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

metrics = [tf.keras.metrics.SparseTopKCategoricalAccuracy(name='accuracy') ]

learning_rate = CustomSchedule() # learning_rate = 0.001  # Instead set a static learning rate
optimizer = tf.keras.optimizers.Adam(learning_rate)

# ### Init Model

model = T5Wrapper.from_pretrained('t5-small')


model.compile(optimizer=optimizer, metrics=metrics)
model.summary()

# +
#Print out structure of the model
# keras.utils.plot_model(model, show_shapes=True, dpi=90)
# -

# ### Start Tensorboard
#

# +
# # %load_ext tensorboard
# # %tensorboard --logdir f"{exp_dir}/tf_data/experiments/t5/logs"
# -

epochs_done = 0
model.fit(tf_train_ds, epochs=epochs, steps_per_epoch=steps,
          validation_data=tf_valid_ds, validation_steps=valid_steps, initial_epoch=epochs_done)

# <hr>

#Reusing the model, just adjusting the experiment
model_path = '/home/ubuntu/karthik/data_speaks_e2e/src/experiments/gen_experiments/model'

# Load Model
model = T5Wrapper.from_pretrained(model_path) #to be uncommented when required. 

# ### Generate Results + Metrics

# +
params_array = []


param1 = {
              'max_length': 45,
              'min_length': 10,
              'early_stopping': True,
              'do_sample': True,
              'temperature': 0.1,
              'top_k': 0,
              'no_repeat_ngram_size': 2
             }

param2 = {
              'max_length': 45,
              'min_length': 10,
              'early_stopping': True,
              'do_sample': True,
              'temperature': 0.7,    
              'top_k': 0,
              'no_repeat_ngram_size': 2
             }


param3 = {
              'max_length': 45,
              'min_length': 10,
              'early_stopping': True,
              'do_sample': True,
              'temperature': 0.5,    
              'top_k': 0,
              'no_repeat_ngram_size': 2
             }

param4 = {
              'max_length': 45,
              'min_length': 10,
              'early_stopping': True,
              'do_sample': True,
              'temperature': 0.3,    
              'top_k': 0,
              'no_repeat_ngram_size': 2
             }

params_array.append(param1)
# params_array.append(param2)
# params_array.append(param3)
# params_array.append(param4)

# max_length (int, optional, defaults to 20) – The maximum length of the sequence to be generated.

# min_length (int, optional, defaults to 10) – The minimum length of the sequence to be generated.

# do_sample (bool, optional, defaults to False) – Whether or not to use sampling ; use greedy decoding otherwise.

# early_stopping (bool, optional, defaults to False) – Whether to stop the beam search when at least num_beams sentences are finished per batch or not.

# num_beams (int, optional, defaults to 1) – Number of beams for beam search. 1 means no beam search.

# temperature (float, optional, defaults tp 1.0) – The value used to module the next token probabilities.

# top_k (int, optional, defaults to 50) – The number of highest probability vocabulary tokens to keep for top-k-filtering.

# top_p (float, optional, defaults to 1.0) – If set to float < 1, only the most probable tokens with probabilities that add up to top_p or higher are kept for generation.

# repetition_penalty (float, optional, defaults to 1.0) – The parameter for repetition penalty. 1.0 means no penalty. See this paper for more details.

# length_penalty (float, optional, defaults to 1.0) – Exponential penalty to the length. 1.0 means no penalty.
# Set to values < 1.0 in order to encourage the model to generate shorter sequences, to a value > 1.0 in order to encourage the model to produce longer sequences.

# no_repeat_ngram_size (int, optional, defaults to 0) – If set to int > 0, all ngrams of that size can only occur once.

# num_return_sequences (int, optional, defaults to 1) – The number of independently computed returned sequences for each element in the batch.

# use_cache – (bool, optional, defaults to True): Whether or not the model should use the past last key/values attentions (if applicable to the model) to speed up decoding.

# +
#Write model outputs

for param_set in params_array:

    #Returns a list of all the model generated outputs
    model_ouput = get_model_output(model, tokenizer, param_set, None, tf_valid_ds, None)

    v_out = model_ouput['validation']['output']
    ts_val=time.strftime("%Y%m%d_%H%M")
    print(ts_val)
    write_model_output(valid_ds, "validation", ts_val, v_out, write_path=exp_dir)
    
    # Let's Use E2E Evaluation Metrics
    scores = compute_metrics(exp_dir, base_dir, ts_val, 'validation', param_set)

    print(scores)
    
    save_metrics(exp_dir, ts_val, scores)    


# +
# Let's Use E2E Evaluation Metrics
scores = compute_metrics(exp_dir, base_dir, ts_val, 'validation', gen_params)

print(scores)
# -

print(scores)
save_metrics(exp_dir, ts_val, scores)



# #### If we like the scores and want to save the scores to our model track
# (We should probably club this with when we save to S3)

# ### Save Model (only if its worth it)


model_path

# Keep for AWS path
model.save_pretrained(f'{model_path}')
# save_model_to_s3(model,base_dir, ts_val)

# ### Load Model

# +
#Below is an optional step to load a pre-trained and saved model to directly run predictions.

#model = T5Wrapper.from_pretrained(model_path) #to be uncommented when required. 
# -

















