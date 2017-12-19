"""
Data generator

"""
import os
import numpy as np
from PIL import Image
from collections import Counter
import random, math
import math
from scipy import misc
import linecache
import ipdb
import torch
from utils import vocab2id, tokenlist2numlist
from collections import defaultdict


class DataGen(object):

    def __init__(self, data_base_dir, data_path, 
        label_path, max_aspect_ratio, max_encoder_l_h, max_encoder_l_w, max_decoder_l):

        # folder with processed images
        self.data_base_dir = data_base_dir 
        # .lst file with name of the image and number
        self.data_path = data_path 
        # .lst file with formulas
        self.label_path = label_path 
        self.max_width = 10000
        self.max_aspect_ratio = max_aspect_ratio
        self.max_encoder_l_h = max_encoder_l_h
        self.max_encoder_l_w = max_encoder_l_w
        self.max_decoder_l = max_decoder_l 
        self.min_aspect_ratio = 0.5
        self.voc2id, self.vocab_size = vocab2id("../data/latex_vocab.txt")

        # create list that will contain the filenames and the line number of the formula
        self.lines = []

        with open(self.data_path, 'r') as file:
            lines_read = file.readlines()

        for line in lines_read:
            filename, label = line.split()
            self.lines.append([filename, label]) 
        
        # buffer to save groups of batches with same width and height
        #self.buffer = {} 
        self.buffer = defaultdict(lambda: defaultdict(list))

    def next_batch(self, batch_size):
        for i in range(0,len(self.lines)):
            cursor = i

            # Get the image path and read the image
            img_path = self.lines[cursor][0] 
            img = misc.imread("../data/images_processed/"+ img_path) 
            # Convert image to grayscale (the shape of the function changes from (h,w,3) to (h,w))
            img = np.average(img, weights = [0.299, 0.587, 0.114], axis = 2)

            # Get the formula number and save it to a list (add start of sequence and end of sequence tokens)
            label_str = self.lines[cursor][1] 
            label_list_tokens = linecache.getline(self.label_path, int(str(label_str))).split() 
            label_list_tokens.insert(0, "<SOS>") 
            label_list_tokens.append("<EOS>") 
            # convert tokens into ids (integers)
            label_list = tokenlist2numlist(label_list_tokens, self.voc2id) 

            origH = img.shape[0]
            origW = img.shape[1]

            # if list of tokens is to big, truncate
            if len(label_list) > self.max_decoder_l: 
                label_list = label_list[:self.max_decoder_l]
 
            # get aspect_ratio and assure is between max and min aspect ratios defined
            bounds_check = (len(label_list), math.floor(origH/8.0), math.floor(origW/8.0))
            bounds_tuple = (self.max_decoder_l, self.max_encoder_l_h, self.max_encoder_l_w)
            if bounds_check <= bounds_tuple:

                aspect_ratio = origW / origH 
                aspect_ratio = min(aspect_ratio, self.max_aspect_ratio) 
                aspect_ratio = max(aspect_ratio, self.min_aspect_ratio)

                imgW = origW 
                imgH = origH 

                if imgW not in self.buffer: 
                    self.buffer[imgW] = {}

                if imgH not in self.buffer[imgW]: 
                    self.buffer[imgW][imgH] = [] 
                
                self.buffer[imgW][imgH].append([img, label_list, img_path])

                # when buffer reaches batch_size, store images and targets in tensors
                if len(self.buffer[imgW][imgH]) == batch_size: 
                    images = torch.Tensor(batch_size, 1, imgH, imgW) 
                    img_paths = []

                    max_target_length = 0 

                    for i in range(len(self.buffer[imgW][imgH])):
                        img_paths.append(self.buffer[imgW][imgH][i][2]) 
                        images[i] = torch.from_numpy(self.buffer[imgW][imgH][i][0]) 
                        max_target_length = max(max_target_length, len(self.buffer[imgW][imgH][i][1])) 

                    targets = np.zeros((batch_size,max_target_length-1))
                    targets = torch.from_numpy(targets)
                    targets_eval = np.zeros((batch_size, max_target_length-1))
                    targets_eval = torch.from_numpy(targets_eval)

                    num_nonzeros = 0
                    for i in range(len(self.buffer[imgW][imgH])): 
                        num_nonzeros = num_nonzeros + len(self.buffer[imgW][imgH][i][1]) - 2 
                        for j in range(len(self.buffer[imgW][imgH][i][1])-1): 
                            targets[i][j] = self.buffer[imgW][imgH][i][1][j]
                            targets_eval[i][j] = self.buffer[imgW][imgH][i][1][j+1]
                            
                    #restart buffer
                    self.buffer[imgW][imgH] = [] 

                    return images, targets, targets_eval, num_nonzeros, img_paths 
