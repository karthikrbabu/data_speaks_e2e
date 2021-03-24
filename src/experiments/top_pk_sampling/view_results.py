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



# ### Load Experiment Results

# +
result = None
path = "output"
exp_count = 1
output_folders = os.listdir(f'{path}')
for folder in output_folders:
    files = os.listdir(f'{path}/{folder}')
    for file in files:
        if file.endswith(".csv"):
            print(f'Exp #{exp_count}: {path}/{folder}/{file}')
            exp_count += 1
            cur_exp = pd.read_csv(f'{path}/{folder}/{file}')
            if result is None:
                cols = list(cur_exp.columns)
                result = pd.DataFrame(columns=cols)
            result = pd.concat([result, cur_exp])

print(result.shape)
resupd.set_option('display.max_colwidth', None)lt.head()
# -

result.sort_values(by='BLEU', ascending=False)


