"""
Utils for training and sampling
"""

import argparse
import os
import numpy
import time
import math


def vocab2id(file_path):
    """
    Creates a dictionary vocab2id based on the file latex_vocab.txt 
    """
    with open(file_path, 'r') as file:
        lines = file.readlines()
        tokens = [line.strip() for line in lines]
        voc2id = {}
        for i in range(len(tokens)):
            voc2id[tokens[i]]=i
        return voc2id

def tokenlist2numlist(line, voc2id):
    """
    Function that converts a list of tokens into its list of ids
    """
    num_list = []
    for token in line:
        num_list.append(voc2id[token])
    return num_list


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))





