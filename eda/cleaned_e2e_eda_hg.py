# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py
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
import math 
import json
import re

import collections
from collections import defaultdict

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
# -

# ### EDA - Clean E2E Dataset

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
# -

#Stopwords Set
sr= set(stopwords.words('english'))


# ### Example of `fixed` vs. `not fixed`

train_mrs=train['meaning_representation']


def get_tags(mr):
    """
    Returns list of tags for a given MR. Also appends to global dictionary to track values for analysis
    mr: single meaning representation
    """
    tag_pairs = mr.split(',')
    
    tags = []
    for pair in tag_pairs:
        tag_name, value = pair.split('[')
        
        tag_name = tag_name.strip()
        value = value[:-1]
        
        tag_values[tag_name].append(value)
        tags.append(tag_name)
    return tags


# +
#Dictionary to track values that tags take on in training
tag_values = defaultdict(lambda: [])

#List of lists, of tags
all_tags = [get_tags(mr) for mr in train_mrs]

#flatten tags_total
flat_tags = [item.strip() for sublist in all_tags for item in sublist]

#Unique Tags
unique_tags = set(flat_tags)

tag_counts = collections.Counter(flat_tags)
tag_counts

# +
# %%time
tag_clean_tokens = []
for token in flat_tags:
    if token not in sr:
        tag_clean_tokens.append(token)

freq_tag = nltk.FreqDist(tag_clean_tokens)
freq_tag.plot(20, cumulative=False)


# -

# ### Understand spread of all words in MR, ignore `[`, `]`, `<any_punctuation>` and `<stopwords>`

# +
# %%time
def get_word_tokens(text_value):
    """
    Returns total list of tokens from the text_value
    text_value: word value that a mr tag may have
    """
    tokens = nltk.word_tokenize(text_value)
    return [word.lower() for word in tokens if word.isalpha()]

#Flatten list, one for each tag key
values = [item for sublist in list(tag_values.values()) for item in sublist] 

#List of lists, of all words
word_list = [get_word_tokens(text) for text in values]

#flatten tags_total
flat_words = [item for sublist in word_list for item in sublist]

#Unique words
unique_words = set(flat_words)

# Word counts
word_counts = collections.Counter(flat_words)
# -

#Non Stop Words
unique_words_non_stop = unique_words - sr


# +
# %%time
word_clean_tokens = []
for token in flat_words:
    if token not in sr:
        word_clean_tokens.append(token)

# Non-Stop Word counts
word_counts_non_stop = collections.Counter(word_clean_tokens)        

word_freq = nltk.FreqDist(word_clean_tokens)
word_freq.plot(20, cumulative=False)
# -
print("Size of MR Vocab (no tags)", len(unique_words))
print("Size of MR Vocab (no tags, no stopwords)", len(word_counts_non_stop))


# <hr>

# ### Shifting focus to the `<ref>` outputs analysis

train_refs = train['human_reference']
train_refs[0]

# +
# %%time
#List of lists, of all words
ref_words = [get_word_tokens(text) for text in train_refs]

#flatten tags_total
flat_ref_words = [item for sublist in ref_words for item in sublist]

#Unique words
unique_ref_words = set(flat_ref_words)

# Word counts
ref_word_counts = collections.Counter(unique_ref_words)

# +
# %%time
ref_word_clean_tokens = []
for token in flat_ref_words:
    if token not in sr:
        ref_word_clean_tokens.append(token)

# Non-Stop Word counts
ref_word_counts_non_stop = collections.Counter(ref_word_clean_tokens)        

word_freq = nltk.FreqDist(ref_word_clean_tokens)
word_freq.plot(20, cumulative=False)
# -

print("Size of Ref Vocab", len(unique_ref_words))
print("Size of Ref Vocab (no stopwords)", len(ref_word_counts_non_stop))




