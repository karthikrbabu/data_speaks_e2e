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

# ### Load Train/ Validation

train = load_dataset('e2e_nlg_cleaned', split='train')
validation = load_dataset('e2e_nlg_cleaned', split='validation')

# ### Init Config

# +
warmup_steps = 1e4
epochs = 5
batch_size = 30
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

class TimeHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.times = []

    def on_epoch_begin(self, batch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, batch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)


# ### Fit Model Variants

# +
#Grid Search

optimizer_list = ['rmsprop','adam', 'adamax'] #'adagrad', 'sgd'] gave poor results
learning_rates = [0.0005, 0.0001, 0.001, 0.005, 'custom']
model_sizes = ['t5-small'] #'t5-base'
special_tokens = [False, True]
encoder_max_lens = [60, 90, 120]
decoder_max_lens = [60, 90, 120]


metrics = [tf.keras.metrics.SparseTopKCategoricalAccuracy(name='accuracy')]

model_variants = []
model_count = 1


# -

def compute(model_size, opt_m, opt, learning_rate, encoder_max_len, decoder_max_len, is_special_tokens):
    """
    Modularize Compute call in case of failures
    """
    global metrics
    global tf_train_ds
    global tf_valid_ds
    global steps
    global valid_steps
    global epochs
    global model_variants
    global model_count
    
    
    print(f"opt: {opt}  learning_rate: {learning_rate} encoder_max_len: {encoder_max_len} decoder_max_len: {decoder_max_len} is_special_tokens:{is_special_tokens}")
    
    if 'model' in globals(): del model

    #Compile Model
    model = T5Wrapper.from_pretrained(model_size)
    model.compile(optimizer=opt_m, metrics=metrics)

    #Model Fit
    epochs_done=0
    time_callback = TimeHistory()
    history_callback = model.fit(tf_train_ds, epochs=epochs, steps_per_epoch=steps, callbacks=[time_callback],
                                validation_data=tf_valid_ds, validation_steps=valid_steps, initial_epoch=epochs_done)

    #Call Backs Data
    times = time_callback.times
    fit_data = history_callback.history

    #Data points 
    total_time = sum(times)
    avg_epoch_time = np.mean(times)

    epoch_num = int(np.argmin(fit_data['val_loss']))
    train_accuracy = np.max(fit_data['accuracy'])
    train_loss = np.min(fit_data['loss'])
    val_accuracy = np.max(fit_data['val_accuracy'])
    val_loss = np.min(fit_data['val_loss'])
    lr_mod = np.min(fit_data['lr'])


    #Gather Details
    model_details = {
        'id': model_count,
        'model_size': model_size,
        'optimizer': opt,
        'learning_rate': learning_rate if type(learning_rate) == float else 'custom',
        'lr_mod': lr_mod,
        'train_accuracy': train_accuracy,
        'train_loss':train_loss,
        'val_accuracy': val_accuracy,
        'val_loss': val_loss,
        'total_time': round(total_time/60, 2),
        'avg_epoch_time': avg_epoch_time,
        'epoch_num': epoch_num,
        'encoder_max_len': encoder_max_len,
        'decoder_max_len': decoder_max_len,
        'is_special_token': is_special_tokens,
       }

    model_variants.append(model_details)


#Execute
for model_size in model_sizes:
    for encoder_max_len in encoder_max_lens:
        for decoder_max_len in decoder_max_lens:    
            for is_special in special_tokens:

                if 'tokenizer' in globals(): del tokenizer
                if 'train_ds' in globals(): del train_ds
                if 'valid_ds' in globals(): del valid_ds
                if 'tf_train_ds' in globals(): del tf_train_ds
                if 'tf_valid_ds' in globals(): del tf_valid_ds

                #Is Special Token
                is_special_tokens = 'yes' if is_special else 'no' 


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


                for lr in learning_rates:
                    for opt in optimizer_list:
                        
                        #### IF SOMETHING BREAKS PICKUP WHERE WE LEFT OFF
                        if model_count < 152:
                            print(f'Skipping: {model_count}')
                            model_count +=1
                            continue
                        ####

                        if lr == 'custom':
                            lr = CustomSchedule()

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
                            print(f"Trying Model {model_count}")
                            compute(model_size, opt_m, opt, lr, encoder_max_len, decoder_max_len, is_special_tokens)
                            model_count +=1
                        except:

                            if len(model_variants):
                                print("WRITING OUT")
                                with open(f'{exp_dir}/exp_chpt/result_{model_count}.txt', 'w') as outfile:
                                    json.dump(model_variants, outfile)
                                traceback.print_exc()
                            else:
                                print("SKIPPING WRITE") 

                try:
        
                    if len(model_variants):
                        print("WRITING OUT")                
                        #Checkpoint ourselves by writing out progress every 15 model trains
                        with open(f'{exp_dir}/exp_chpt/result_{model_count}.txt', 'w') as outfile:
                            json.dump(model_variants, outfile)
                    else:
                        print("SKIPPING WRITE") 
                except:
                    print("ERROR WITH CYCLIC WRITE")
                    traceback.print_exc()







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
ts_val=time.strftime("%Y%m%d_%H%M")
print(ts_val)
write_model_output(valid_ds, "validation", ts_val, v_out, write_path=exp_dir)


# Let's Use E2E Evaluation Metrics
scores = compute_metrics(exp_dir, base_dir, ts_val, 'validation', gen_params)
print(scores)

print(scores)
save_metrics(exp_dir, ts_val, scores)

# #### If we like the scores and want to save the scores to our model track
# (We should probably club this with when we save to S3)

# ### Save Model (only if its worth it)


# Keep for AWS path
model.save_pretrained(f'{model_path}')
# save_model_to_s3(model,base_dir, ts_val)

# ### Load Model

# +
#Below is an optional step to load a pre-trained and saved model to directly run predictions.

#model = T5Wrapper.from_pretrained(model_path) #to be uncommented when required. 