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
from transformers import (TFAutoModelWithLMHead, AutoTokenizer, PreTrainedModel,
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

# !pwd

# ### Setup Directories

#AWS box path we should keep
# data_dir = "/home/ubuntu/praveen/data_speaks_e2e/tf_data"
data_dir = "/home/karthikrbabu/data_speaks_e2e/tf_data"
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

# ### Data Pipeline

def encode(example, encoder_max_len=encoder_max_len, decoder_max_len=decoder_max_len):
    """
    Encode function that uses the T5 Tokenizer on each example
    """
    mr = example['meaning_representation']
    ref = example['human_reference']
  
    mr_base = f"data_to_text: {str(mr)}"
    ref_base = f"{str(ref)}"

    encoder_inputs = tokenizer(mr_base, truncation=True, 
                               return_tensors='tf', max_length=encoder_max_len,
                              pad_to_max_length=True)

    decoder_inputs = tokenizer(ref_base, truncation=True, 
                               return_tensors='tf', max_length=decoder_max_len,
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


def create_dataset(dataset, cache_path=None, batch_size=batch_size, 
                   buffer_size= buffer_size, shuffling=True):
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



tf_train_ds= create_dataset(tf_train_ds, batch_size=batch_size, 
                         shuffling=True, cache_path = None)
tf_valid_ds = create_dataset(tf_valid_ds, batch_size=batch_size, 
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
# %tensorboard --logdir /home/ubuntu/praveen/data_speaks_e2e/tf_data/experiments/t5/logs

epochs_done = 0
model.fit(tf_train_ds, epochs=1, steps_per_epoch=steps, callbacks=callbacks, 
          validation_data=tf_valid_ds, validation_steps=valid_steps, initial_epoch=epochs_done)

import time
ts=time.strftime("%Y%m%d_%H%M")
print(ts)

# ### Save Model

# Keep for AWS path
# model.save_pretrained(f'/home/ubuntu/praveen/data_speaks_e2e/model_runs/{ts}/')
model.save_pretrained(f'/home/karthikrbabu/data_speaks_e2e/model_runs/{ts}/')



# ### Load Model

loaded_model = T5Wrapper.from_pretrained(f'/home/karthikrbabu/data_speaks_e2e/model_runs/{ts}/')

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


# ### Working on batch generation

# +
def get_model_output(model, tok, tf_train_ds=None, tf_valid_ds=None, tf_test_ds=None):
    """
    Once model is trained and saved. Use this function to evaluate the output on train, validation, and test 
    
    model => TFPretrainedModel - T5Wrapper base class
    tok => AutoTokenizer - Tokenizer used for the respective model
    tf_train_ds => PreFetchDataset - of Batched Tensors (train)
    tf_valid_ds => PreFetchDataset - of Batched Tensors (validation)
    tf_test_ds => PreFetchDataset - of Batched Tensors (test)
    
    """
    
    def gen_output(ds, ds_name):
        """
        ds => PreFetchDataset  - of Batched Tensors
        ds_name => String - Name of Dataset
        
        """
        
        print(f"Starting {ds_name}")
        start = time.time()
        output = []
        ds_iter = iter(list(ds))
        
        isNext = True
        while isNext:
            input_batch = next(ds_iter, None)
            if input_batch:
                input_batch.pop('labels', None)
                input_batch.pop('decoder_attention_mask', None)

                hypotheses_batch = model.generate(
                    **input_batch,
                    num_beams=4,
                    length_penalty=2.0,
                    max_length=142,
                    min_length=56,
                    no_repeat_ngram_size=3,
                    do_sample=False,
                    early_stopping=True,
                )
                decoded = tok.batch_decode(hypotheses_batch, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                output += decoded
            else:
                isNext = False

        timestamp2 = time.time()
        print("Took %.2f seconds" % ((timestamp2 - timestamp1)))
        print("Took {0} minutes".format((timestamp2 - timestamp1)/60))
        print()
        return output

    
    train_output = gen_output(tf_train_ds, "Train") if tf_train_ds else []
    validation_output = gen_output(tf_valid_ds, "Validation") if tf_valid_ds else []
    test_output = gen_output(tf_test_ds, "Test") if tf_test_ds else []
    
    return {"train_output": train_output, "validation_output":validation_output, "test_output":test_output}
    
    
# -

def write_out_tsv(hf_ds, hf_ds_name, sys_out):
    """
    Write out TSV file once we have gotten respective datasets model generated outputs
    
    hf_ds => HuggingFaceDataset - features: ['meaning_representation', 'human_reference']
    gen_out => List - Corresponding model generated outputs for hf_ds
    hf_ds_name => String - Name of Dataset
    
    """

    print(f"Writing {hf_ds_name}_out.csv")
    source = hf_ds['meaning_representation']
    reference = hf_ds['human_reference']
    system_output = sys_out
    df = pd.DataFrame({"source": source, "reference":reference, "output":system_output})
    
    df.to_csv(f'{hf_ds_name}_out.tsv', sep='\t', header=True, index=False)
    




