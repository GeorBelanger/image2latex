"""
Utils for training and sampling
"""

import argparse
import os
import numpy
import time
import math
import linecache
import datetime

def timeSince(since, percent):
    """
    Function that calculates the time spent and estimated time left for training 
    based on the percentage of iterations processed so far
    """
    now = time.time()
    s = now - since
    # estimated total time
    es = s / (percent)
    # estimated time remaining
    rs = es - s
    return '%s (- %s)' % (str(datetime.timedelta(seconds=s)), str(datetime.timedelta(seconds=rs)))

class Tokenizer():
    def __init__(self):
        self.vocab2id = {"<SOS>":0, "<EOS>":1}
        self.id2vocab = ["<SOS>", "<EOS>"]

    def add_token_to_vocab(self, token):
        if token not in self.vocab2id:
            self.id2vocab.append(token)
            self.vocab2id[token]=len(self.id2vocab)-1
        return self.vocab2id[token]

    def tokenize(self, path, line_number):
        label_list_tokens = linecache.getline(path, int(str(line_number))).split()
        label_list_tokens.insert(0, "<SOS>") 
        label_list_tokens.append("<EOS>")

        ids = []
        for token in label_list_tokens:
            self.add_token_to_vocab(token)
            ids.append(self.vocab2id[token])
        return ids

def vocab2id(file_path):
    """
    Creates a dictionary vocab2id based on the file latex_vocab.txt 
    """
    with open(file_path, 'r') as file:
        lines = file.readlines()
        tokens = [line.strip() for line in lines]
        voc2id = {}
        vocab_size = len(tokens)
        for i in range(len(tokens)):
            voc2id[tokens[i]]=i
        return voc2id, vocab_size





