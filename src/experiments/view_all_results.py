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
import json
import pandas as pd
import numpy as np
import os

pd.set_option('display.max_colwidth', None)
# -

all_exp_folders = [x[0] for x in os.walk('./') if 'output/2021' in x[0]]
all_exp_folders[:5]

# ### Load Experiment Results

# +
result = None
exp_count = 1
for folder in all_exp_folders:
    files = os.listdir(f'{folder}')
    for file in files:
        if file.endswith(".csv"):
#             print(f'Exp #{exp_count}: {folder}/{file}')
            exp_count += 1
            cur_exp = pd.read_csv(f'{folder}/{file}')
            if result is None:
                cols = list(cur_exp.columns)
                result = pd.DataFrame(columns=cols)
            result = pd.concat([result, cur_exp])

print(result.shape)
result.head()
# -

result.sort_values(by='BLEU', ascending=False)


