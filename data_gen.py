import os
import numpy as np
from PIL import Image
from collections import Counter
import math
from scipy import misc
import ipdb
import torch
from utils import vocab2id, Tokenizer, read_formulas_directory
from collections import defaultdict
import imageio


class DataLoader(object):
    """Load the images and labels from the database and process into batches
    Attributes:
        data_base_dir (str): Folder with the processed images
        label_path (str): File with latex math formulas
        max_aspect_ratio (int): Maximum aspect ratio (width/height) for images
        max_encoder_l_h (int): Maximum size for the images height
        max_encoder_l_w (int): Maximum size for the images width
        max_decoder_l (int): Maximum number of tokens for the latex formula
    """

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
        self.vocab_size = 1000
        self.tokenizer = Tokenizer()
        # buffer to save groups of batches with same width and height
        self.buffer = defaultdict(lambda: defaultdict(list))

    def process_batch(self, buf, img_width, img_height):
        """ Return a batch of images with labels and take it out of the buffer
        Args:
            buf (:obj:dict:dict:list): object containing images according
                                       to the images size and width
            img_width (int): size of the image's width in the batch
            img_height (int): size of the image's height in the batch
        """
        # store images and targets in tensors
        batch_size = len(buf[img_width][img_height])
        images = torch.Tensor(batch_size, 1, img_height, img_width)
        img_paths = []
        max_target_length = max([len(buf_element[1]) for buf_element
                                 in buf[img_width][img_height]])

        for k in range(batch_size):
            img_paths.append(buf[img_width][img_height][k][2])
            images[k] = torch.from_numpy(buf[img_width][img_height][k][0])

        targets = torch.zeros(batch_size, max_target_length-1)
        targets_eval = torch.zeros(batch_size, max_target_length-1)

        num_nonzer = 0
        for m in range(len(buf[img_width][img_height])):
            num_nonzer = (num_nonzer +
                          len(buf[img_width][img_height][m][1]) - 2)
            for j in range(len(buf[img_width][img_height][m][1])-1):
                targets[m][j] = buf[img_width][img_height][m][1][j]
                targets_eval[m][j] = buf[img_width][img_height][m][1][j+1]
        # restart buffer
        buf[img_width][img_height] = []
        return images, targets, targets_eval, num_nonzer, img_paths

    def create_data_generator(self, batch_size, directory_path):
        """ Create a generator that will yield the images and labels
        Args:
            batch_size (int): size of the batch to generate
            directory_path (string): path of the file containing
                                     filenames of the images and formulas
        """
        image_list = read_formulas_directory(directory_path)

        for i in range(0, len(image_list)):
            # Get the image path and read the image
            img_path = image_list[i][0]
            img = imageio.imread("../data/images_processed/" + img_path)
            # Convert color image to grayscale
            # (the shape of the image object changes from (h,w,3) to (h,w))
            rgb2gray_weights = [0.299, 0.587, 0.114]
            img = np.average(img, weights=rgb2gray_weights, axis=2)
            # Get the formula number and save it to a list
            label_str = image_list[i][1]
            # tokenize function
            label_list = self.tokenizer.tokenize(self.label_path, label_str)

            origH = img.shape[0]
            origW = img.shape[1]

            # if list of tokens is too big, truncate
            if len(label_list) > self.max_decoder_l:
                label_list = label_list[:self.max_decoder_l]

            bounds_check = (len(label_list), math.floor(origH/8.0),
                            math.floor(origW/8.0))
            bounds_tuple = (self.max_decoder_l, self.max_encoder_l_h,
                            self.max_encoder_l_w)
            if bounds_check <= bounds_tuple:
                # get aspect_ratio and assure is between
                # max and min aspect ratios defined
                aspect_ratio = origW / origH
                aspect_ratio = min(aspect_ratio, self.max_aspect_ratio)
                aspect_ratio = max(aspect_ratio, self.min_aspect_ratio)

                imgW = origW
                imgH = origH

                self.buffer[imgW][imgH].append([img, label_list, img_path])
                # when buffer reaches batch_size,
                # return images and targets as tensors
                if len(self.buffer[imgW][imgH]) == batch_size:
                    images, targets, targets_eval, num_nonzer, img_paths = (
                     self.process_batch(self.buffer, imgW, imgH))
                    yield images, targets, targets_eval, num_nonzer, img_paths

                # when we have gone through all the lines,
                # return incomplete batches stored in buffer
                if i == len(image_list)-1:
                    for imgW in self.buffer:
                        for imgH in self.buffer[imgW]:
                            if len(self.buffer[imgW][imgH]) > 0:
                                images, targets, targets_eval, num_nonzer, img_paths = (
                                 self.process_batch(self.buffer, imgW, imgH))
                                yield images, targets, targets_eval, num_nonzer, img_paths
