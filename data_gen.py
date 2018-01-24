"""
Data generator

"""
import os
import numpy as np
from PIL import Image
from collections import Counter
from random import shuffle
import math
from scipy import misc
import ipdb
import torch
from utils import vocab2id, Tokenizer
from collections import defaultdict


class DataLoader(object):

    def __init__(self, 
        data_base_dir, 
        label_path, 
        max_aspect_ratio, 
        max_encoder_l_h, 
        max_encoder_l_w, 
        max_decoder_l):

        # folder with processed images
        self.data_base_dir = data_base_dir 
        # .lst file with formulas
        self.label_path = label_path 
        self.max_width = 10000
        self.max_aspect_ratio = max_aspect_ratio
        self.max_encoder_l_h = max_encoder_l_h
        self.max_encoder_l_w = max_encoder_l_w
        self.max_decoder_l = max_decoder_l 
        self.min_aspect_ratio = 0.5
        self.voc2id, self.vocab_size = vocab2id("../data/latex_vocab.txt")
        self.tokenizer = Tokenizer()

        # buffer to save groups of batches with same width and height
        self.buffer = defaultdict(lambda: defaultdict(list))

    def read_directory(self, path):
        # create list that will contain the filenames and the line number of the formula
        with open(path, 'r') as file:
            lines_read = file.readlines()
            image_list = [list(line.split()) for line in lines_read] 
            shuffle(image_list)
            return image_list

    def process_batch(self, buf, img_width, img_height):
        # store images and targets in tensors
        batch_size = len(buf[img_width][img_height])
        images = torch.Tensor(batch_size, 1, img_height, img_width) 
        img_paths = []

        max_target_length = 0 

        for k in range(batch_size):
            img_paths.append(buf[img_width][img_height][k][2]) 
            images[k] = torch.from_numpy(buf[img_width][img_height][k][0]) 
            max_target_length = max(max_target_length, len(buf[img_width][img_height][k][1])) 

        targets = torch.zeros(batch_size, max_target_length-1)
        targets_eval = torch.zeros(batch_size, max_target_length-1)

        num_nonzeros = 0
        for m in range(len(buf[img_width][img_height])): 
            num_nonzeros = num_nonzeros + len(buf[img_width][img_height][m][1]) - 2 
            for j in range(len(buf[img_width][img_height][m][1])-1): 
                targets[m][j] = buf[img_width][img_height][m][1][j]
                targets_eval[m][j] = buf[img_width][img_height][m][1][j+1]        
        #restart buffer
        buf[img_width][img_height] = [] 
        return images, targets, targets_eval, num_nonzeros, img_paths
        
    def create_data_generator(self, batch_size, directory_path):
        image_list = self.read_directory(directory_path)

        for i in range(0,len(image_list)):
            # Get the image path and read the image
            img_path = image_list[i][0] 
            img = misc.imread("../data/images_processed/"+ img_path) 
            # Convert image to grayscale 
            #(the shape of the function changes from (h,w,3) to (h,w))
            img = np.average(img, weights = [0.299, 0.587, 0.114], axis = 2)
            # Get the formula number and save it to a list 
            label_str = image_list[i][1] 
            #tokenize function
            label_list = self.tokenizer.tokenize(self.label_path, label_str)

            origH = img.shape[0]
            origW = img.shape[1]

            # if list of tokens is too big, truncate
            if len(label_list) > self.max_decoder_l:
                label_list = label_list[:self.max_decoder_l]

            bounds_check = (len(label_list), math.floor(origH/8.0), math.floor(origW/8.0))
            bounds_tuple = (self.max_decoder_l, self.max_encoder_l_h, self.max_encoder_l_w)
            if bounds_check <= bounds_tuple:
                # get aspect_ratio and assure is between max and min aspect ratios defined
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

                # when buffer reaches batch_size, return images and targets in tensors
                if len(self.buffer[imgW][imgH]) == batch_size: 
                    images, targets, targets_eval, num_nonzeros, img_paths = self.process_batch(self.buffer, imgW, imgH)
                    yield images, targets, targets_eval, num_nonzeros, img_paths

                # when we have gone through all the lines, return incomplete batches stored in buffer
                if i == len(image_list)-1:
                    images, targets, targets_eval, num_nonzeros, img_paths = self.process_batch(self.buffer, imgW, imgH)
                    yield images, targets, targets_eval, num_nonzeros, img_paths
