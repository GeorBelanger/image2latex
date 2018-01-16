"""
Utils for training and sampling
"""

import argparse
import os
import numpy
import time
import math
import linecache


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







