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
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

# -

# ### Datasets

# +
#e2e_nlg
#e2e_nlg_cleaned

dataset = load_dataset("e2e_nlg_cleaned")
train = dataset['train']
dev = dataset['validation']
test = dataset['test']
print("Train Size", train.shape)
print("Dev Size", dev.shape)
print("Test Size", test.shape)

train

# +
train = pd.read_csv('../data/data_sandbox/train_senti.csv')
dev = pd.read_csv('../data/e2e-cleaning-master/cleaned-data/devel-fixed.no-ol.csv') 
test = pd.read_csv('../data/e2e-cleaning-master/cleaned-data/test-fixed.csv') 
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

train.to_csv('/home/karthikrbabu/data_speaks_e2e/data/data_sandbox/train_senti.csv')
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


