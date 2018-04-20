"""Utils for training and sampling"""

import argparse
import os
import numpy
import time
import math
import linecache
import datetime
from random import shuffle
import torch
import ipdb


def timeSince(since, percent):
    """Calculate the time spent and
    estimated time left for training based on the
    percentage of iterations processed so far"""
    now = time.time()
    s = now - since
    # estimated total time
    es = s / (percent)
    # estimated time remaining
    rs = es - s
    return '%s (- %s)' % (str(datetime.timedelta(seconds=s)),
                          str(datetime.timedelta(seconds=rs)))


class Tokenizer():
    """ Construct the dictionary of different tokens encountered in the math
    formulas used for training"""
    def __init__(self, initial_id2voc, initial_voc2id):
        self.vocab2id = initial_voc2id
        self.id2vocab = initial_id2voc

    def add_token_to_vocab(self, token):
        if token not in self.vocab2id:
            self.id2vocab.append(token)
            self.vocab2id[token] = len(self.id2vocab)-1
        return self.vocab2id[token]

    def tokenize(self, path, line_number):
        label_list_tokens = linecache.getline(path,
                                              int(str(line_number))).split()
        label_list_tokens.insert(0, "<SOS>")
        label_list_tokens.append("<EOS>")

        ids = []
        for token in label_list_tokens:
            self.add_token_to_vocab(token)
            ids.append(self.vocab2id[token])
        return ids


def vocab2id(file_path):
    """Create a dictionary vocab2id based on the file latex_vocab.txt"""
    with open(file_path, 'r') as file:
        lines = file.readlines()
        tokens = [line.strip() for line in lines]
        voc2id = {}
        vocab_size = len(tokens)
        for i in range(len(tokens)):
            voc2id[tokens[i]] = i
        return voc2id, vocab_size


def read_formulas_directory(path):
    """ Create a list that will contain the filenames and the line number of the
    math formulas used for training"""
    with open(path, 'r') as file:
        lines_read = file.readlines()
        image_list = [list(line.split()) for line in lines_read]
        shuffle(image_list)
        return image_list


def make_one_hot_vector_from_index(index, vocab_size):
    one_hot = numpy.zeros((vocab_size, 1))
    one_hot[index] = 1
    return one_hot


def tokens_from_index_list(index_list, id2vocab):
    """Given a list of index, get the respective token list"""
    token_list = []
    for i in range(len(index_list)):
        if index_list[i] > len(id2vocab)-1:
            token_list.append("<UNK>")
        else:
            token_list.append(id2vocab[index_list[i]])
    return token_list


def slice_as_longtensor(targets, index):
    """Given a batch of targets, returns a slice of it as
    a torch longtensor """
    sliced_targets = targets.narrow(1, index, 1)
    sliced_targets_long = torch.LongTensor(sliced_targets.numpy().astype(int))
    return sliced_targets_long
