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

# ### Indicate in the below cell, whether you want to look at all experiments ~ OR~ just a single experiment

experiment_name = 'all'

# +
if not experiment_name or experiment_name == 'all':
    exp_folders = [x[0] for x in os.walk('./') if 'output/2021' in x[0]]
else:
    exp_folders = [x[0] for x in os.walk(f'./{experiment_name}') if 'output/2021' in x[0]]

print("Total Experiments: ", len(exp_folders))
print()
print("Sample:")
print(np.array(exp_folders[:5]))

# -

# ### Load Experiment Results

# +
result = None
exp_count = 1
for folder in exp_folders:
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
# -

result.sort_values(by='BLEU', ascending=False).head(20)




