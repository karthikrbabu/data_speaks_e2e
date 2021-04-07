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

# # Final Model Iteration
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
import traceback
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
import keras

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

# ### Load Train/ Validation

# +
train = load_dataset('e2e_nlg_cleaned', split='train')
validation = load_dataset('e2e_nlg_cleaned', split='validation')

validation
# -

# ### Init Config

# +
fall_back_epochs = 5
batch_size = 30
buffer_size = 1000
ntrain = len(train)
nvalid = len(validation)
steps = int((ntrain//fall_back_epochs)// batch_size)
valid_steps = int((nvalid//fall_back_epochs)// batch_size)

print("Train Data Length: ", ntrain)
print("Validation Data Length: ", nvalid)
print("Total Steps: ", steps)
print("Total Validation Steps: ", valid_steps)
print("Batch Size: ", batch_size)
print("Total fall_back_epochs: ", fall_back_epochs)
# -

# ### Model Variants and Param Sets
#

# +
path = f'{base_dir}/src/experiments/model_training_experiment/exp_chpt/result_271.txt'
with open(path) as json_file:
    models_data = json.load(json_file)
    
path = f'{exp_dir}/param_sets.txt'
with open(path) as json_file:
    param_sets = json.load(json_file)
# -

# ### Model Variants

# +
praveen = [172, 247, 157]
karthik = [67, 82]

id_set = set(karthik)
model_variants = [model for model in models_data if model['id'] in id_set]
print("Number of variants: ", len(model_variants))
# -

# ### Param Sets

print("Number of param_sets: ", len(param_sets))


class TimeHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.times = []

    def on_epoch_begin(self, batch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, batch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)


# +
def compute_generation(model, tokenizer):
    global tf_valid_ds
    global validation
    global exp_dir
    global base_dir
    global param_sets
    
    print("Starting Generate Variants:")
    param_count = 1
    for param_set in param_sets:
        
#         ### IF SOMETHING BREAKS PICKUP WHERE WE LEFT OFF
#         if param_count < PICK_THE_NUM:
#             print(f'Skipping: Model#: {model_count} Param#: {param_count}')
#             param_count +=1
#             continue
#         ###
        
        print(f"Generate {param_count}/{len(param_sets)}")
        print(str(param_set))

        #Returns a list of all the model generated outputs
        model_ouput = get_model_output(model, tokenizer, param_set, None, tf_valid_ds, None)

        v_out = model_ouput['validation']['output']
        ts_val=time.strftime("%Y%m%d_%H%M")
        print(ts_val)
        write_model_output(validation, "validation", ts_val, v_out, write_path=exp_dir)

        # Let's Use E2E Evaluation Metrics
        scores = compute_metrics(exp_dir, base_dir, ts_val, 'validation', param_set)

        print(scores)
        print()
        save_metrics(exp_dir, ts_val, scores)
        param_count +=1
        


# -

def compute_model(model_size, opt_m, opt, learning_rate, encoder_max_len, decoder_max_len, epoch_num, tokenizer):
    """
    Modularize Compute call in case of failures
    """
    global metrics
    global tf_train_ds
    global tf_valid_ds
    global steps
    global valid_steps
    global model_count
    global fall_back_epochs
    
    print(f"Computing Model ===> {model_count}")

    if 'model' in globals(): del model

    #Compile Model
    model = T5Wrapper.from_pretrained(model_size)
    model.compile(optimizer=opt_m, metrics=metrics)
    
    #Handle epoch_num
    ep = fall_back_epochs if epoch_num == "NONE" else epoch_num

    #Model Fit
    epochs_done=0
    time_callback = TimeHistory()
    
    history_callback = model.fit(tf_train_ds, epochs=ep, steps_per_epoch=steps, callbacks=[time_callback],
                                validation_data=tf_valid_ds, validation_steps=valid_steps, initial_epoch=epochs_done)

    #Call Backs Data
    times = time_callback.times

    #Data points 
    total_time = sum(times)
    print(f"Model Training Time: {total_time}")

    compute_generation(model, tokenizer)




# ## Kick it Off! 

# +
metrics = [tf.keras.metrics.SparseTopKCategoricalAccuracy(name='accuracy')]
model_count = 1

for train_params in model_variants:

    if 'tokenizer' in globals(): del tokenizer
    if 'train_ds' in globals(): del train_ds
    if 'valid_ds' in globals(): del valid_ds
    if 'tf_train_ds' in globals(): del tf_train_ds
    if 'tf_valid_ds' in globals(): del tf_valid_ds

    opt = train_params['optimizer']
    lr = train_params['lr_mod']
    epoch_num = train_params.get('epoch_num', "NONE")
    is_special_token = train_params['is_special_token']
    encoder_max_len = train_params['encoder_max_len']
    decoder_max_len = train_params['decoder_max_len']
    model_size = train_params['model_size']
    
    print(f"Model {model_count}/{len(model_variants)} opt: {opt}  lr: {lr} epoch_num: {epoch_num} encoder_max_len: {encoder_max_len} decoder_max_len: {decoder_max_len} is_special_token:{is_special_token}")
    
    #Is Special Token
    is_special = True if is_special_token =='yes' else False
    
    ### Init Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_size, additional_special_tokens=['data_to_text:']) if is_special else AutoTokenizer.from_pretrained(model_size)

    ### Process Train/ Validation
    train_ds = train.map(lambda x: encode(x, tokenizer, False, encoder_max_len=encoder_max_len, decoder_max_len=decoder_max_len))
    valid_ds = validation.map(lambda x: encode(x, tokenizer, False, encoder_max_len=encoder_max_len, decoder_max_len=decoder_max_len))

    ### Process Train/Validation =>  Tensors
    tf_train_ds = to_tf_dataset(train_ds)
    tf_valid_ds = to_tf_dataset(valid_ds)

    ### Build Train/ Validation =>  Model Ready Input
    tf_train_ds= create_dataset(tf_train_ds, batch_size=batch_size, 
                     shuffling=True, cache_path = None)
    tf_valid_ds = create_dataset(tf_valid_ds, batch_size=batch_size, 
                     shuffling=False, cache_path = None)

    #### IF SOMETHING BREAKS PICKUP WHERE WE LEFT OFF
#     if model_count < 152:
#         print(f'Skipping: {model_count}')
#         model_count +=1
#         continue
    ####

    if opt == 'rmsprop':
        opt_m = tf.keras.optimizers.RMSprop(lr)

    elif opt == 'adam':
        opt_m = tf.keras.optimizers.Adam(lr)

    elif opt == 'adagrad':
        opt_m = tf.keras.optimizers.Adagrad(lr)

    elif opt == 'adamax':
        opt_m = tf.keras.optimizers.Adamax(lr)

    elif opt == 'sgd':
        opt_m = tf.keras.optimizers.SGD(lr)

    try:
        compute_model(model_size, opt_m, opt, lr, encoder_max_len, decoder_max_len, epoch_num, tokenizer)
        model_count +=1
    except:
        print(f"Failed on: Model#{model_count}")
        traceback.print_exc()

# -





