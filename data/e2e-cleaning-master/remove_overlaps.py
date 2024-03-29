#!/usr/bin/env python3

import re
from argparse import ArgumentParser

import pandas as pd

from tgen.data import DA


def parse_mr(mr_text):
    return DA.parse_diligent_da(mr_text).get_delexicalized(set(['name', 'near']))

def main(args):

    train = pd.read_csv(args.input_train, encoding="UTF-8")
    train['mr'] = train['mr'].fillna('')
    devel = pd.read_csv(args.input_devel, encoding="UTF-8")
    devel['mr'] = devel['mr'].fillna('')
    test = pd.read_csv(args.input_test, encoding="UTF-8")
    test['mr'] = test['mr'].fillna('')

    test_orig_mrs = set([parse_mr(mr) for mr in list(test['orig_mr'])])
    test_mrs = set([parse_mr(mr) for mr in list(test['mr'])])
    print("Test set distinct MR count: %d, originally %d" % (len(test_mrs), len(test_orig_mrs)))

    print("Checking devel set...")

    # check devel data:
    devel_idx_to_del = []
    devel_orig_mrs = set([parse_mr(mr) for mr in list(devel['orig_mr'])])
    devel_mrs = set()
    avoid = test_mrs | test_orig_mrs
    for idx, inst in devel.iterrows():
        mr = parse_mr(inst.mr)
        devel_mrs.add(mr)
        if mr in avoid:
            devel_idx_to_del.append(idx)

    print("To delete: %d / %d instances from devel, %d / %d distinct MRs" %
          (len(devel_idx_to_del), len(devel), len(devel_mrs & avoid), len(devel_mrs)))
    devel = devel.drop(devel_idx_to_del)

    output_devel = re.sub('(\.[^.]+)$', args.suffix + r'\1', args.input_devel)
    print("Writing fixed %s..." % output_devel)
    devel.to_csv(output_devel, encoding='UTF-8', index=False)
    print("%d instances, %d distinct (delexicalized) MRs." % (len(devel), len(devel_mrs - avoid)))
    print("Original distinct MR count: %d" % len(devel_orig_mrs))


    print("Checking train set...")

    # check train data
    train_idx_to_del = []
    train_orig_mrs = set([parse_mr(mr) for mr in list(train['orig_mr'])])
    train_mrs = set()
    avoid = devel_mrs | test_mrs | test_orig_mrs
    for idx, inst in train.iterrows():
        mr = parse_mr(inst.mr)
        train_mrs.add(mr)
        if mr in avoid:
            train_idx_to_del.append(idx)

    print("To delete: %d / %d instances from devel, %d / %d distinct MRs" %
          (len(train_idx_to_del), len(train), len(train_mrs & avoid), len(train_mrs)))
    train = train.drop(train_idx_to_del)

    output_train = re.sub('(\.[^.]+)$', args.suffix + r'\1', args.input_train)
    print("Writing fixed %s..." % output_train)
    train.to_csv(output_train, encoding='UTF-8', index=False)
    print("%d instances, %d distinct (delexicalized) MRs." % (len(train), len(train_mrs - avoid)))
    print("Original distinct MR count: %d" % len(train_orig_mrs))



if __name__ == '__main__':
    ap = ArgumentParser(description='Remove overlapping MRs from different parts of the dataset (aiming to keep the test set intact)')
    ap.add_argument('--suffix', '-s', type=str, default='.no-ol', help='Suffix to add to output filenames')
    ap.add_argument('input_train', type=str, help='Input training set CSV')
    ap.add_argument('input_devel', type=str, help='Input development set CSV')
    ap.add_argument('input_test', type=str, help='Input test set CSV')
    args = ap.parse_args()

    main(args)

