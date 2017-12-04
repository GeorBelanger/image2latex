"""
Data generator

This file contains the object DataGen. 

The key arguments is receives are the following:
data_base_dir, which is a folder with the processed images
label_path, the path of a .lst file with the latex formulas, one per line
data_path, the path of a .lst file with the name of the image and its corresponding line number

The main functions are the initialization, shuffling and next_batch function

"""
import os
import numpy as np
from PIL import Image
from collections import Counter
import random, math
import math
from random import shuffle
from scipy import misc
import linecache
import ipdb
import torch
from utils import vocab2id, tokenlist2numlist


class DataGen(object):

    def __init__(self, data_base_dir, data_path, 
        label_path, max_aspect_ratio, max_encoder_l_h, max_encoder_l_w, max_decoder_l):

        self.data_base_dir = data_base_dir #folder with processed images
        self.data_path = data_path # .lst file with name of the image and number
        self.label_path = label_path # .lst file with formulas
        self.max_width = 10000
        self.max_aspect_ratio = max_aspect_ratio
        self.max_encoder_l_h = max_encoder_l_h
        self.max_encoder_l_w = max_encoder_l_w
        self.max_decoder_l = max_decoder_l # in the lua paper were like this max_decoder_l or math.inf
        self.max_target_length = -10000 # they set it up to -math.huge in the paper (makes no sense) (doesnt affect because is actually makes the max be the )
        self.min_aspect_ratio = 0.5
        self.voc2id = vocab2id("../data/latex_vocab.txt")

        #add log (pending)

        #create list of lines
        self.lines = []

        #Open file with name of image and number of formula
        with open(self.data_path, 'r') as file:
            lines_read = file.readlines()

        #Create index to count lines read and input dictionary of lines
        idx = 0 
        for line in lines_read:
            idx = idx + 1
            filename, label = line.split()
            self.lines.append([filename, label])  #self.lines is a list that contains lists of filenames and labels

        self.cursor = 0 # cursor to know in which line are we
        self.buffer = {} # buffer to save groups of batches with same width and height
        


    def shuffle(self):
        shuffle(self.lines)

    def next_batch(self, batch_size):
        while True:
            if self.cursor > len(self.lines)-1: # if we have gone through all the lines, finish
                break
            # Get the image path and read the image
            img_path = self.lines[self.cursor][0] # get the image path
            img = misc.imread("../data/images_processed/"+ img_path) #Read image
            # Convert image to grayscale (the shape of the function changes from (h,w,3) to (h,w))
            img = np.average(img, weights = [0.299, 0.587, 0.114], axis = 2)


            #Get the formula number and save it to a list
            label_str = self.lines[self.cursor][1] # get the label number 
            #ipdb.set_trace()
            label_list_tokens = linecache.getline(self.label_path, int(str(label_str))).split() # save the formula to a list
            label_list_tokens.insert(0, "<SOS>") # add start of sequence token to the beginning of the list
            label_list_tokens.append("<EOS>") # add end of sequence token to the end of the list
            label_list = tokenlist2numlist(label_list_tokens, self.voc2id) #convert tokens into ids (integers)
            
            self.cursor = self.cursor + 1 # go to next line

            origH = img.shape[0]
            origW = img.shape[1]

            if len(label_list) > self.max_decoder_l: # if list of tokens is to big, truncate
                temp = [] # temporal table
                for i in range(self.max_decoder_l): # save the truncated tokens in a temporal list
                    temp.append(label_list[i])
                label_list = temp # new list truncated

            if len(label_list)<= self.max_decoder_l and math.floor(origH/8.0) <= self.max_encoder_l_h and math.floor(origW/8.0) <= self.max_encoder_l_w:
                aspect_ratio = origW / origH # get aspect_ratio and assure is between max and min aspect ratios defined
                aspect_ratio = min(aspect_ratio, self.max_aspect_ratio) 
                aspect_ratio = max(aspect_ratio, self.min_aspect_ratio)

                imgW = origW # get image width
                imgH = origH # get image height

                if self.buffer.get(imgW) == None: # if width is not in buffer, create a new dictionary
                    self.buffer[imgW] = {}

                if self.buffer[imgW].get(imgH) == None: 
                    self.buffer[imgW][imgH] = [] # if width and height is not in buffer, create a new list.
                
                self.buffer[imgW][imgH].append([img, label_list, img_path]) #insert to the buffer a list with the image, the list of tokens and the img_path

                if len(self.buffer[imgW][imgH]) == batch_size: # when buffer reaches batch_size
                    #ipdb.set_trace()
                    images = torch.Tensor(batch_size, 1, imgH, imgW) # create tensor to store images
                    img_paths = []

                    max_target_length = self.max_target_length # set the max_target_length for the sequences in the batch

                    for i in range(len(self.buffer[imgW][imgH])):
                        img_paths.append(self.buffer[imgW][imgH][i][2]) # save image path
                        images[i] = torch.from_numpy(self.buffer[imgW][imgH][i][0]) #save the images into the tensor
                        max_target_length = max(max_target_length, len(self.buffer[imgW][imgH][i][1])) # max_target_length based on the number of tokens

                    # targets: use as input. SOS, ch1, ch2, ..., chn 
                    #targets = torch.IntTensor(batch_size, max_target_length-1)
                    targets = np.zeros((batch_size,max_target_length-1))
                    targets = torch.from_numpy(targets)
                    #targets_eval: use for evaluation. ch1, ch2, ..., chn, EOS 
                    #targets_eval = torch.IntTensor(batch_size, max_target_length-1)
                    targets_eval = np.zeros((batch_size, max_target_length-1))
                    targets_eval = torch.from_numpy(targets_eval)

                    num_nonzeros = 0
                    for i in range(len(self.buffer[imgW][imgH])): # for every element in the actual buffer
                        num_nonzeros = num_nonzeros + len(self.buffer[imgW][imgH][i][1]) - 2 # acumulate the number of tokens
                        #ipdb.set_trace()
                        for j in range(len(self.buffer[imgW][imgH][i][1])-1): # save every token and the following one
                            targets[i][j] = self.buffer[imgW][imgH][i][1][j]
                            targets_eval[i][j] = self.buffer[imgW][imgH][i][1][j+1]
                            

                    self.buffer[imgW][imgH] = [] #restart buffer

            # Still missing what happens when image is bigger than expected

                    return images, targets, targets_eval, num_nonzeros, img_paths 

         
            
#datagen = DataGen("../data/images_processed", "../data/im2latex_train_filter.lst", "../data/im2latex_formulas.norm.lst", 1000, 1000, 1000, 1000)
#datagen.shuffle()
#images, targets, targets_eval, num_nonzeros, img_paths = datagen.next_batch(5)

