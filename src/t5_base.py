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

#Custom Utils Lib
from utils.utils import (get_model_output, write_pre_metrics_data, add_model_record,
                         encode, to_tf_dataset, create_dataset, compute_metrics, save_model_to_s3)
from classes.t5Wrapper import T5Wrapper
from classes.customScheduler import CustomSchedule

#AWS
import boto3
s3 = boto3.resource('s3')


# +
tf_version = tf.__version__
print("Tensorflow: ", tf_version)
print("Transformers: ", transformers.__version__)
print("Datasets: ", datasets.__version__)

tf_version_split = tf_version.split('.')
assert int(tf_version_split[0])==2 and int(tf_version_split[-2])>=3, f"Tensorflow version should be '2.3+,x', given {tf_version}"

# -

# ### Setup Directories

# +
#AWS box path we should keep
# data_dir = "/home/ubuntu/praveen/data_speaks_e2e/tf_data"
base_dir = os.path.abspath(os.path.join(os.getcwd(),os.pardir))
data_dir = f"{base_dir}/tf_data"
log_dir = f"{data_dir}/experiments/t5/logs"
save_path = f"{data_dir}/experiments/t5/models"
cache_path_train = f"{data_dir}/cache/t5.train"
cache_path_test = f"{data_dir}/cache/t5.test"

print("base directory: ",base_dir)

ts_val=time.strftime("%Y%m%d_%H%M")
model_path = f'{base_dir}/model_runs/ts={ts_val}/model'
model_gen_out_path = f'{base_dir}/model_runs/ts={ts_val}'
metrics_path = base_dir + '/e2e-metrics-master'

print('model_path: ', model_path)
print('model_gen_out_path: ', model_gen_out_path)
print('metrics_path: ', metrics_path)
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
encoder_max_len = 60
decoder_max_len = 60
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

train_ds = train.map(lambda x: encode(x, tokenizer))
valid_ds = validation.map(lambda x: encode(x, tokenizer))

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

learning_rate = CustomSchedule() # learning_rate = 0.001  # Instead set a static learning rate
optimizer = tf.keras.optimizers.Adam(learning_rate)

# ### Init Model

model = T5Wrapper.from_pretrained('t5-small')


model.compile(optimizer=optimizer, metrics=metrics)
model.summary()

# ### Start Tensorboard
#

# %load_ext tensorboard
# %tensorboard --logdir f"{base_dir}/tf_data/experiments/t5/logs"

epochs_done = 0
model.fit(tf_train_ds, epochs=1, steps_per_epoch=steps, callbacks=callbacks, 
          validation_data=tf_valid_ds, validation_steps=valid_steps, initial_epoch=epochs_done)

# <hr>

# ### Generate Results + Metrics

# +
gen_params = {'num_beams': 1, 
              'max_length': 60,
              'min_length': 20, 
              'early_stopping': True,
              'do_sample': False, 
              'no_repeat_ngram_size': 2 
             }

#Returns a list of all the model generated outputs
model_ouput = get_model_output(model, tokenizer, {}, None, tf_valid_ds, None)
# -
#Write model outputs
v_out = model_ouput['validation']['output']
write_pre_metrics_data(valid_ds, "validation", v_out, write_path=model_gen_out_path)


# Let's Use E2E Evaluation Metrics
scores = compute_metrics(model_gen_out_path, metrics_path, ds_name='validation')
scores

# #### If we like the scores and want to save the scores to our model track
# (We should probably club this with when we save to S3)

# +
# add_model_record(base_dir, scores)
# -

# ### Save Model

# Keep for AWS path
# model.save_pretrained(f'/home/ubuntu/praveen/data_speaks_e2e/model_runs/{ts}/')
model.save_pretrained(f'{base_dir}/model_runs/{ts_val}/')

save_model_to_s3(model, ts_val)

# ### Load Model

loaded_model = T5Wrapper.from_pretrained(model_path)



loaded_model = T5Wrapper.from_pretrained(f'{base_dir}/model_runs/{ts}/')
mr = validation['meaning_representation'][200]
print(mr)

input_text =  f"data_to_text: {mr}"
print(input_text)
encoded_query = tokenizer(input_text, 
                         return_tensors='tf', pad_to_max_length=True, truncation=True, max_length=encoder_max_len)
input_ids = encoded_query["input_ids"]
attention_mask = encoded_query["attention_mask"]
print(input_ids)
generated_answer = loaded_model.generate(input_ids, attention_mask=attention_mask, 
                                 max_length=decoder_max_len, top_p=0.95, top_k=50, repetition_penalty=2)
decoded_answer = tokenizer.decode(generated_answer.numpy()[0])
print("Model REF: ", decoded_answer)


