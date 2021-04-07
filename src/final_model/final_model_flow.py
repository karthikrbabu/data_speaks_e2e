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
# The purpose of this notebook is to showcase the best model and generation experiment we have achieved through our research.
#
# #### Steps:
# * Import Libraries
# * Load Train/Dev/Test Data
# * Config Definitions
# * Pre-process Data (Tensors)
# * Quick Tensor EDA
# * Optimizer Init
# * Train TF T5 on E2E Cleaned Data to Text Problem
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

base_dir = os.path.abspath(os.path.join(os.getcwd(),os.pardir, os.pardir))
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

# ### Load Train + Validation + Test

# +
train = load_dataset('e2e_nlg_cleaned', split='train')
validation = load_dataset('e2e_nlg_cleaned', split='validation')
test = load_dataset('e2e_nlg_cleaned', split='test')

test
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

path = f'{base_dir}/src/experiments/model_training_experiment/exp_chpt/result_271.txt'
with open(path) as json_file:
    models_data = json.load(json_file)


# ### Model Variants
#
# From all the variants we have tried, we found that model ID `82` gave us the best performance. Please refer to the notebook found in `data_speaks_e2e/src/experiments/model_training_experiment` to learn more.

best_model_id = 82
model_variants = [model for model in models_data if model['id'] == best_model_id]
model_variants

# ### Param Sets

param_sets = [ 
    {'num_beams': 5, 
     'max_length': 220, 
     'min_length': 20, 
     'early_stopping': False, 
     'do_sample': False, 
     'no_repeat_ngram_size': 2},

    {'num_beams': 5,
     'max_length': 45, 
     'min_length': 10, 
     'early_stopping': False, 
     'do_sample': False, 
     'no_repeat_ngram_size': 2}
]

print("Number of param_sets: ", len(param_sets))


class TimeHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.times = []

    def on_epoch_begin(self, batch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, batch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)


def compute_generation(model, tokenizer):
    global tf_valid_ds
    global tf_test_ds
    global validation
    global test
    global exp_dir
    global base_dir
    global param_sets
    
    print("Starting Generate Variants:")
    param_count = 1
    for param_set in param_sets:

        print(f"Generate {param_count}/{len(param_sets)}")
        print(str(param_set))

        #Returns a dictionary for model output for test 
        model_ouput = get_model_output(model, tokenizer, param_set, None, None, tf_test_ds)

        #Test Out
        test_out = model_ouput['test']['output']
        ts_val=time.strftime("%Y%m%d_%H%M")
        print(ts_val)
        write_model_output(test, "test", ts_val, test_out, write_path=exp_dir)

        # Let's Use E2E Evaluation Metrics
        scores = compute_metrics(exp_dir, base_dir, ts_val, 'test', param_set)

        print(scores)
        print()
        save_metrics(exp_dir, ts_val, scores)
        
        
        param_count +=1


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
    global model
    
    print(f"Computing Model ===> {model_count}")

    if 'model' not in globals():
        
        print("TRAINING A MODEL")
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

    else:
        print("GRABBING THE MODEL")

    #This is computed both on the validation and test data
    compute_generation(model, tokenizer)
    





# ## Kick it Off! 

train_params = model_variants[0]
train_params

# +
metrics = [tf.keras.metrics.SparseTopKCategoricalAccuracy(name='accuracy')]
model_count = 1

####
# Using already trained model
model = T5Wrapper.from_pretrained(f'{exp_dir}/model')
###

#Grab all Model details from config
opt = train_params['optimizer']
lr = train_params['lr_mod']
epoch_num = train_params.get('epoch_num', "NONE")
is_special_token = train_params['is_special_token']
encoder_max_len = train_params['encoder_max_len']
decoder_max_len = train_params['decoder_max_len']
model_size = train_params['model_size']

print(f"Model {model_count} opt: {opt}  lr: {lr} epoch_num: {epoch_num} encoder_max_len: {encoder_max_len} decoder_max_len: {decoder_max_len} is_special_token:{is_special_token}")

#Is Special Token
is_special = True if is_special_token =='yes' else False

### Init Tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_size, additional_special_tokens=['data_to_text:']) if is_special else AutoTokenizer.from_pretrained(model_size)

### Process Train/ Validation
train_ds = train.map(lambda x: encode(x, tokenizer, False, encoder_max_len=encoder_max_len, decoder_max_len=decoder_max_len))
valid_ds = validation.map(lambda x: encode(x, tokenizer, False, encoder_max_len=encoder_max_len, decoder_max_len=decoder_max_len))
test_ds = test.map(lambda x: encode(x, tokenizer, False, encoder_max_len=encoder_max_len, decoder_max_len=decoder_max_len))

### Process Train/Validation =>  Tensors
tf_train_ds = to_tf_dataset(train_ds)
tf_valid_ds = to_tf_dataset(valid_ds)
tf_test_ds = to_tf_dataset(test_ds)

### Build Train/ Validation =>  Model Ready Input
tf_train_ds= create_dataset(tf_train_ds, batch_size=batch_size, 
                 shuffling=True, cache_path = None)
tf_valid_ds = create_dataset(tf_valid_ds, batch_size=batch_size, 
                 shuffling=False, cache_path = None)
tf_test_ds = create_dataset(tf_test_ds, batch_size=batch_size, 
                 shuffling=False, cache_path = None)

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
    print(f"Failed on: Model: #{model_count}")
    traceback.print_exc()
# -











