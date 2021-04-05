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

# # Data Sentiment Analysis
#
# Goal of this notebook is to use the pre-built HuggingFace Distill-BERT sentiment analysis pipeline on our training and validation data. We did this to understand if there was any implicit bias in our data that would affect our model to generate text that would be predominantly positive or negative.

# +
#Utilities
import pandas as pd
import numpy as np
from scipy import stats
import math 
import json
import re
import collections
from collections import defaultdict


#Timing
import timeit
import datetime
import time

#Plotting
import matplotlib.pyplot as plt
import plotly.express as px

#NLTK 
from nltk.tokenize import WordPunctTokenizer
from nltk.tokenize import wordpunct_tokenize
from nltk.tokenize import word_tokenize 
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords

import nltk

#HuggingFace
from datasets import list_datasets, load_dataset
from transformers import pipeline, AutoTokenizer

# -

# ### Datasets

# +
train = pd.read_csv('../../data/data_sandbox/train_senti.csv')
dev = pd.read_csv('../../data/e2e-cleaning-master/cleaned-data/devel-fixed.no-ol.csv') 
test = pd.read_csv('../../data/e2e-cleaning-master/cleaned-data/test-fixed.csv') 
print("Train Size", train.shape)
print("Dev Size", dev.shape)
print("Test Size", test.shape)

train = train.drop(["Unnamed: 0"], axis = 1)
dev = dev.drop(["fixed", "orig_mr"], axis = 1)
test = test.drop(["fixed", "orig_mr"], axis = 1)


train.head()
# -

# ### Sentiment of Training Data

#By default this uses 'distilbert-base-uncased-finetuned-sst-2-english'
#https://huggingface.co/transformers/model_doc/distilbert.html
classifier = pipeline('sentiment-analysis')

print(train.shape)
train.head()

# ### Add Sentiment column to Training Data

# +
train_refs = list(train['ref'])

sent_labels = []
sent_scores = []

step = 50
for lb in range(0,len(train['ref']), step):
    ub = lb + step
    if ub > 33525:
        ub = 33525
    
    start_time = time.time()
    sentiments = classifier(train_refs[lb:ub])
    print(f"{lb}-{ub}: --- {(time.time() - start_time)} seconds ---")
    
    for x in sentiments:
        sent_labels.append(x['label'])
        sent_scores.append(x['score'])

train['labels'] = sent_labels
train['scores'] = sent_scores

train.to_csv('/home/ubuntu/karthik/data_speaks_e2e/data/data_sandbox/train_senti.csv')
# +
result = train.groupby('labels').agg({'scores': ['mean', 'min', 'max','count', 'std','var']}).reset_index()
print("NEGATIVE %: ", 12912/(12912+20613))
print("POSITIVE %: ", 20613/(12912+20613))

negative_scores = list(train[train['labels'] == 'NEGATIVE']['scores'])
positive_scores = list(train[train['labels'] == 'POSITIVE']['scores'])

print()
print(f'POSITIVE stats: {stats.describe(positive_scores)}')
print()
print(f'NEGATIVE stats: {stats.describe(negative_scores)}')

result


# +
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(8, 5))
axes[0].hist(negative_scores, density=True, alpha=0.5, label='NEGATIVE') # density=False would make counts
axes[1].hist(positive_scores, density=True, alpha=0.5, label='POSITIVE')
fig.tight_layout()


axes[0].legend(loc='upper right')
axes[1].legend(loc='upper right')

plt.show()
# -
# ## Train Conclusions:
#
# Our given human references in the training data have this sentiment spread, bias towards positivie summaries.
# * NEGATIVE %:  0.3851454138702461 (12,912 count)
# * POSITIVE %:  0.614854586129754 (20,613 count)
#
#
# Mostly very confident in the sentiments, negative class is marginally less confident. 
# * POSITIVE stats: DescribeResult(nobs=20613, minmax=(0.5003801584243774, 0.999880313873291), mean=0.9650171614020567, variance=0.0069442250681093914, skewness=-3.478629338147737, kurtosis=12.225515875040173)
#
#
# * NEGATIVE stats: DescribeResult(nobs=12912, minmax=(0.5000444650650024, 0.9998053312301636), mean=0.9516976303253623, variance=0.009525433403789625, skewness=-2.770861182607165, kurtosis=7.26381817812592)

# <hr>


print(dev.shape)
dev.head()

# ### Add Sentiment column to Validation Data

# +
dev_refs = list(dev['ref'])

sent_labels = []
sent_scores = []

step = 50
for lb in range(0,len(dev['ref']), step):
    ub = lb + step
    if ub > 4299:
        ub = 4299
    
    start_time = time.time()
    sentiments = classifier(dev_refs[lb:ub])
    print(f"{lb}-{ub}: --- {(time.time() - start_time)} seconds ---")
    
    for x in sentiments:
        sent_labels.append(x['label'])
        sent_scores.append(x['score'])

dev['labels'] = sent_labels
dev['scores'] = sent_scores

dev.to_csv('/home/ubuntu/karthik/data_speaks_e2e/data/data_sandbox/dev_senti.csv')

# +
result = dev.groupby('labels').agg({'scores': ['mean', 'min', 'max','count', 'std','var']}).reset_index()
print("NEGATIVE %: ", 12912/(12912+20613))
print("POSITIVE %: ", 20613/(12912+20613))

negative_scores = list(dev[dev['labels'] == 'NEGATIVE']['scores'])
positive_scores = list(dev[dev['labels'] == 'POSITIVE']['scores'])

print()
print(f'POSITIVE stats: {stats.describe(positive_scores)}')
print()
print(f'NEGATIVE stats: {stats.describe(negative_scores)}')

result

# +
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(8, 5))
axes[0].hist(negative_scores, density=True, alpha=0.5, label='NEGATIVE') # density=False would make counts
axes[1].hist(positive_scores, density=True, alpha=0.5, label='POSITIVE')
fig.tight_layout()


axes[0].legend(loc='upper right')
axes[1].legend(loc='upper right')

plt.show()
# -






