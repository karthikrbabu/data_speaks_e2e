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

# ## EDA - Clean E2E Dataset

# +
train = pd.read_csv('../data/e2e-cleaning-master/cleaned-data/train-fixed.no-ol.csv')
dev = pd.read_csv('../data/e2e-cleaning-master/cleaned-data/devel-fixed.no-ol.csv') 
test = pd.read_csv('../data/e2e-cleaning-master/cleaned-data/test-fixed.csv') 
print("Train Size", train.shape)
print("Dev Size", dev.shape)
print("Test Size", test.shape)

train.head()
# -

#Stopwords Set
sr= set(stopwords.words('english'))


# ### Example of `fixed` vs. `not fixed`

#Not Fixed
print("Not Fixed:")
print(train.loc[0]['mr'])
print(train.loc[0]['orig_mr'])
print()
#Fixed
print("Fixed:")
print(train.loc[1]['mr'])
print(train.loc[1]['orig_mr'])

# ## Meaning Representations 
#
# ### Understand spread of MR Tags

train_mrs=list(train['mr'])


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

print("Total # of Tags: ", f'{len(flat_tags):,}')

tag_counts = collections.Counter(flat_tags)
tag_counts
# -

# #### Tags Barchart

# +
df_tag_counts = pd.DataFrame.from_dict(dict(tag_counts), orient='index', dtype=None, columns=['count'])
df_tag_counts['tag'] = df_tag_counts.index

fig = px.bar(df_tag_counts, x='tag', y='count',
             hover_data=['tag', 'count'], color='tag',
             labels={'count':'number of tags in training data'}, height=400)

fig.update_layout(title_text="Training: Tag Count")
fig.show()


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

#Word Counts, no stop words
words_nonstop = []
for token in flat_words:
    if token not in sr:
        words_nonstop.append(token)

# Non-Stop Word counts
word_counts_non_stop = collections.Counter(words_nonstop)

# +
df_mr_word_counts = pd.DataFrame.from_dict(dict(word_counts_non_stop), orient='index', dtype=None, columns=['count']) \
                                .sort_values(by=['count'], ascending=False)
df_mr_word_counts['word'] = df_mr_word_counts.index

fig = px.bar(df_mr_word_counts, x='word', y='count',
             hover_data=['word', 'count'], color='word',
             labels={'count':'word count'}, height=400)

fig.update_layout(title_text="Training: MR Word Count")
fig.show()
# -

print("Size of MR Vocab (no tags)", len(unique_words))
print("Size of MR Vocab (no tags, no stopwords)", len(word_counts_non_stop))


# ### Train MR Lengths

# +
train_mrs_lengths = pd.DataFrame({'lengths': [len(text) for text in train_mrs]})
print(stats.describe(train_mrs_lengths['lengths']))
fig = px.histogram(train_mrs_lengths, x="lengths", histnorm='percent') # histnorm: percent, probability, density

fig.update_layout(title_text="Histogram: Train MR Lengths")
fig.show()
# -

# ### Train MR Token Counts
#
# We want to see an upper bound here, so we are doing no filtering. Stop words and punctuation are left as is

train_mrs_token_counts = pd.DataFrame({'counts': [len(nltk.word_tokenize(text)) for text in train_mrs]})
print(stats.describe(train_mrs_token_counts['counts']))
fig = px.histogram(train_mrs_token_counts, x="counts", histnorm='percent') # histnorm: percent, probability, density
fig.update_layout(title_text="Histogram: Train MR Token Counts")
fig.show()

# <hr>

# ## Human References
# ### Shifting focus to the `<ref>` outputs analysis

train_refs = train['ref']
train_refs[0]

# +
# %%time
#List of lists, of all words
ref_words = [get_word_tokens(text) for text in train_refs]

#flatten tags_total
flat_ref_words = [item for sublist in ref_words for item in sublist]

# Word counts
ref_word_counts = collections.Counter(flat_ref_words)

ref_word_nonstop = []
for token in flat_ref_words:
    if token not in sr:
        ref_word_nonstop.append(token)

# Non-Stop Word counts
ref_word_counts_non_stop = collections.Counter(ref_word_nonstop)        


# +
df_ref_word_counts = pd.DataFrame.from_dict(dict(ref_word_counts_non_stop), orient='index', dtype=None, columns=['count']) \
                                .sort_values(by=['count'], ascending=False)
df_ref_word_counts['word'] = df_ref_word_counts.index
print("Shape: ", df_ref_word_counts.shape)

#Top 150 rows captures a lot of the data
fig = px.bar(df_ref_word_counts[:150], x='word', y='count',
             hover_data=['word', 'count'], color='word',
             labels={'count':'word count'}, height=400)

fig.update_layout(title_text="Training: Ref Word Count")
fig.show()
# -

print("Size of Ref Vocab", len(ref_word_counts))
print("Size of Ref Vocab (no stopwords)", len(ref_word_counts_non_stop))

# ### Train Human Reference Lengths

# +
train_refs_lengths = pd.DataFrame({'lengths': [len(text) for text in train_refs]})
print(stats.describe(train_refs_lengths['lengths']))
fig = px.histogram(train_refs_lengths, x="lengths", histnorm='percent') # histnorm: percent, probability, density

fig.update_layout(title_text="Histogram: Train Ref Lengths")
fig.show()
# -

# ### Train Ref Token Counts
#
# We want to see an upper bound here, so we are doing no filtering. Stop words and punctuation are left as is

train_refs_token_counts = pd.DataFrame({'counts': [len(nltk.word_tokenize(text)) for text in train_refs]})
print(stats.describe(train_refs_token_counts['counts']))
fig = px.histogram(train_refs_token_counts, x="counts", histnorm='percent') # histnorm: percent, probability, density
fig.update_layout(title_text="Histogram: Train Ref Token Counts")
fig.show()

# ## Conclusions:
#
# ### MRs:
# * Vocab Size: 106
# * Vocab Size (no tags, no stopwords) 98
# * Somewhat even spread of all the unique tags (total 8) 
# * MR Lengths: minmax=(13, 210), mean=102.7668903803132
# * MR Token Counts: minmax=(4, 61), mean=29.216405667412378
# * Lengths and counts have a normal distribution! 
#
# ### Refs:
# * Vocab Size: 2020
# * Vocab Size (no tags, no stopwords) 1910
# * Ref Lengths: minmax=(4, 343), mean=111.51633109619686
# * Ref Token Counts: minmax=(1, 73), mean=22.2082624906786
# * Lengths and counts have a normal distribution! 


