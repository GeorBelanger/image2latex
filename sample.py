"""Training script"""

import os
import time
import numpy as np
import argparse
from data_gen import DataLoader
import ipdb
import torch
import torch.nn as nn
from torch.autograd import Variable
from random import shuffle

from model import CNN
from model import EncoderBRNN
from model import AttnDecoderRNN
from model import Attn
import token_dictionary

from utils import timeSince, tokens_from_index_list, slice_as_longtensor

use_cuda = torch.cuda.is_available()

# Create argument parser
parser = argparse.ArgumentParser()

# Data paths
parser.add_argument('--data_base_dir', type=str,
                    default='../data/images_processed',
                    help='Path of folder with processed images')
parser.add_argument('--label_path', type=str,
                    default='../data/im2latex_formulas.norm.lst',
                    help='Path of file with the latex formulas, one per line')
parser.add_argument('--train_path', type=str,
                    default="../data/im2latex_train_filter.lst",
                    help='Path of train file with the name of the image and its \
                          corresponding line number')
parser.add_argument('--validate_path', type=str,
                    default="../data/im2latex_validate_filter.lst",
                    help='Path of validate file with the name of the image and its \
                          corresponding line number')
parser.add_argument('--vocabulary', type=str,
                    default='../data/latex_vocab.txt',
                    help='Path of file with the latex vocab, one per line')
parser.add_argument('--save_cnn', type=str,  default='cnn_model.pt',
                    help='path to save the CNN model')
parser.add_argument('--save_encoder', type=str,  default='encoder_model.pt',
                    help='path to save the BRNN encoder model')
parser.add_argument('--save_decoder', type=str,  default='decoder_model.pt',
                    help='path to save the attention decoder model')

# Max and mins for data generator
parser.add_argument('--max_aspect_ratio', type=float,
                    default=10, help='Maximum permited aspect ratio of images')
parser.add_argument('--max_encoder_l_h', type=float, default=20,
                    help='Maximum permited size for the image height')
parser.add_argument('--max_encoder_l_w', type=float, default=64,
                    help='Maximum permited size for the image width')
parser.add_argument('--max_decoder_l', type=float, default=150,
                    help='Maximum permited size (number of tokens) for the \
                          associated latex formula')
parser.add_argument('--max_vocab_size', type=float, default=560,
                    help='Maximum number of tokens in vocabulary')
# Hyperparameters
parser.add_argument('--num_epochs', type=int, default=20,
                    help='Number of epochs for training')
parser.add_argument('--batch_size', type=int, default=5,
                    help='Batch size for training')
parser.add_argument('--batch_size_eval', type=int, default=1,
                    help='Batch size for evaluation')
parser.add_argument('--learning_rate', type=float, default=0.001,
                    help='Initial learning rate for training')
parser.add_argument('--gradient_clip', type=float, default=1000.0,
                    help='Initial learning rate for training')
# Encoder
parser.add_argument('--num_layers_encoder', type=int, default=1,
                    help='Number of layers in the bidirectional recurrent \
                          neural network used for encoding')
parser.add_argument('--hidden_dim_encoder', type=int, default=512)
parser.add_argument('--max_length_encoder', type=int, default=1000)

# Decoder
parser.add_argument('--output_dim_decoder', type=int, default=1000)
parser.add_argument('--num_layers_decoder', type=int, default=1)
parser.add_argument('--max_length_decoder', type=int, default=1000)
parser.add_argument('--embedding_size', type=int, default=80)

# parse arguments
args = parser.parse_args()


def train(images, targets, targets_eval, cnn, encoder, decoder, cnn_optimizer,
          encoder_optimizer, decoder_optimizer, gradient_clip, criterion,
          max_length, use_cuda):
    """ Makes the forward pass through the cnn, encoder and decoder and then
    calculates the loss, backpropagates and updates the parameters
    Arguments:
      :images: tensor of images with dims (batch_sizeX1XheightXwidth)
      :targets: tensor with target labels for input with
      dims (batch_sizeXmax_length_of_batch) (<SOS>, token1, ... tokenN)
      :targets_eval: tensor with target labels for evaluation with dims
      (batch_sizeXmax_length_of_batch) (token1, ..., tokenN, <EOS>)
      :cnn, encoder, decoder: modules of model
      :cnn_optimizer, encoder_optimizer, decoder_optimizer: model optimizers
      :gradient_clip: (double) max norm of the gradients
      :criterion: criterion for measuring the loss function
      :max_length: (int) maximum sequence length for encoder outputs
      :use_cuda: (boolean) indicates whether the model is using a GPU
    Returns:
      loss per token (double)
      predicted index (list of predicted tokens)
      actual index (list of correct index)

    """
    cnn_optimizer.zero_grad()
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    batch_size = images.size(0)
    target_length = targets.size(1)

    images = Variable(images)
    images = images.cuda() if use_cuda else images

    loss = 0

    # Forward Pass
    # Convolutional network
    list_rows = cnn(images)

    # Encoder
    outputs = encoder(list_rows)

    # Decoder
    encoder_outputs = Variable(torch.zeros(max_length, batch_size,
                               encoder.hidden_dim_encoder))
    encoder_outputs = encoder_outputs.cuda() if use_cuda else encoder_outputs

    # The encoder output should be inside the encoder_outputs of size
    # (max_length, batch_size, hidden_dim)
    encoder_outputs[:outputs.size(0)] = outputs

    # Calculate the first hidden vector of the decoder LSTM
    (decoder_hidden, decoder_cell_state) = decoder.init_hidden_cell(batch_size, encoder.hidden_dim_encoder)
    decoder_context_output = decoder.init_context_output(batch_size, encoder.hidden_dim_encoder)

    # save values for evaluation
    predicted_index = []
    actual_index = []

    for di in range(target_length):

        # Take the di-th target for each batch
        decoder_input = Variable(slice_as_longtensor(targets, di))
        decoder_input = decoder_input.cuda() if use_cuda else decoder_input

        # Take the di-th target_eval for each batch
        decoder_eval = Variable(slice_as_longtensor(targets_eval,
                                                    di).squeeze())
        decoder_eval = decoder_eval.cuda() if use_cuda else decoder_eval

        (decoder_output,
         decoder_context_output,
         decoder_hidden,
         decoder_cell_state) = decoder(decoder_input,
                                       decoder_context_output,
                                       decoder_hidden,
                                       decoder_cell_state,
                                       encoder_outputs)
        loss += criterion(decoder_output, decoder_eval)

        # save values for evaluation

        if use_cuda:
            predicted_index.append(torch.max(decoder_output.data, 1)[1][0])
            actual_index.append(decoder_eval.data[0])
        else:
            predicted_index.append(torch.max(decoder_output.data, 1)[1][0])
            actual_index.append(decoder_eval.data[0])

    loss.backward()
    torch.nn.utils.clip_grad_norm(cnn.parameters(), gradient_clip)
    torch.nn.utils.clip_grad_norm(encoder.parameters(), gradient_clip)
    torch.nn.utils.clip_grad_norm(decoder.parameters(), gradient_clip)
    cnn_optimizer.step()
    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.data[0]/target_length, predicted_index, actual_index


def evaluate(images, targets, targets_eval, cnn, encoder, decoder, criterion,
             max_length, use_cuda):
    """ Makes the forward pass through the cnn and encoder and goes through the
    decoder using beam search to get predictions of the tokens given the image
    Arguments:
      :images: tensor of images with dims (batch_sizeX1XheightXwidth)
      :targets: tensor with target labels for input with
      dims (batch_sizeXmax_length_of_batch) (<SOS>, token1, ... tokenN)
      :targets_eval: tensor with target labels for evaluation with dims
      (batch_sizeXmax_length_of_batch) (token1, ..., tokenN, <EOS>)
      :cnn, encoder, decoder: modules of model
      :criterion: criterion for measuring the loss function
      :max_length: (int) maximum sequence length for encoder outputs
      :use_cuda: (boolean) indicates whether the model is using a GPU
    Returns:
      predicted index (list of predicted tokens)
      actual index (list of correct index)
    """

    images = Variable(images)
    images = images.cuda() if use_cuda else images
    batch_size = images.size(0)
    target_length = targets.size(1)

    # Forward Pass
    # Convolutional network
    list_rows = cnn(images)

    # Encoder
    outputs = encoder(list_rows)

    # Decoder
    encoder_outputs = Variable(torch.zeros(max_length, batch_size,
                               encoder.hidden_dim_encoder))
    encoder_outputs = encoder_outputs.cuda() if use_cuda else encoder_outputs

    # The encoder output should be inside the encoder_outputs of size
    # (max_length, batch_size, hidden_dim)
    encoder_outputs[:outputs.size(0)] = outputs

    # Calculate the first hidden vector of the decoder LSTM
    (decoder_hidden, decoder_cell_state) = decoder.init_hidden_cell(batch_size, encoder.hidden_dim_encoder)
    decoder_context_output = decoder.init_context_output(batch_size, encoder.hidden_dim_encoder)

    # teacher forcing: feed the target as the next input
    # save values for evaluation
    predicted_index = []
    actual_index = []

    # start decoder_input with SOS_token that is already in targets
    decoder_input = Variable(slice_as_longtensor(targets, 0))
    decoder_input = decoder_input.cuda() if use_cuda else decoder_input

    beam_size = 5
    # first pass of decoder
    (initial_output,
     initial_context_output,
     initial_hidden,
     initial_cell_state) = decoder(decoder_input,
                                   decoder_context_output,
                                   decoder_hidden,
                                   decoder_cell_state,
                                   encoder_outputs)

    # take the top elements (most probable)
    top_beam_size = torch.topk(initial_output, beam_size)
    sequences = []
    # loop through the probable elements
    for i in range(0, beam_size):
        # append the top index, log_probability, hidden and cell
        sequences.append(([top_beam_size[1][0].data[i]],
                          -top_beam_size[0][0].data[i],
                          initial_hidden,
                          initial_cell_state,
                          initial_context_output))

    # sequences will look like: [([token_index_for_seq_element_1,
    #                              token_index_for_seq_element_2, ...],
    #                              beam_heuristic_for_sequence,
    #                              last_hidden, last_cell, last_context_output]

    # we have our starting elements already in sequences
    for decoder_input_index in range(max_length-1):
        new_sequences = []
        for beam_index in range(0, beam_size):
            # get the last element of the sequence we just appended to
            associated_sequence = sequences[beam_index][0]
            associated_sequence_score = sequences[beam_index][1]

            if associated_sequence[-1] == 1:
                new_sequences.append(sequences[beam_index])
            else:
                decoder_input = Variable(torch.LongTensor([[associated_sequence[-1]]]))
                decoder_input = decoder_input.cuda() if use_cuda else decoder_input

                last_hidden = sequences[beam_index][2]
                last_cell = sequences[beam_index][3]
                last_context_output = sequences[beam_index][4]

                # decoder_output is [prob_for_vocab_element_1, prob_for_vocab_element_2, etc.]
                (decoder_output,
                 decoder_context_output,
                 decoder_hidden,
                 decoder_cell_state) = decoder(decoder_input,
                                               last_context_output,
                                               last_hidden,
                                               last_cell,
                                               encoder_outputs)

                # Take the top 5 most probable vocab tokens
                top_beam_size = torch.topk(decoder_output, beam_size)

                for j in range(0, beam_size):
                    new_sequences.append((associated_sequence + [top_beam_size[1][0].data[j]],
                                          associated_sequence_score*-top_beam_size[0][0].data[j],
                                          decoder_hidden, decoder_cell_state,
                                          decoder_context_output))

        # maybe these have to go out of the loop
        sorted_new_sequences = sorted(new_sequences, key=lambda x: x[1])
        sequences = sorted_new_sequences[:5]

        # Take the target_eval for each batch
        if decoder_input_index < target_length:
            decoder_eval = Variable(slice_as_longtensor(targets_eval,
                                                        decoder_input_index))
            decoder_eval = decoder_eval.cuda() if use_cuda else decoder_eval

            if use_cuda:
                actual_index.append(decoder_eval.data[0])
            else:
                actual_index.append(decoder_eval.data[0])

    return sequences[0][0], actual_index


def trainIters(num_epochs, batch_size, cnn, encoder, decoder, data_loader,
               data_loader_eval, learning_rate, gradient_clip,
               n_iters, print_every, use_cuda,
               evaluate_with_beam_search=False):
    """ Starts a timer, initialize optimizers and criterion, creates the data
    generators and then repeatedly calls the train function. Next, it prints
    the average loss and the predicted and actual tokens for an image.
    If evaluate_with_beam_search is True, it calls the evaluate function to
    evaluate using beam search.
    Arguments:
      :num_epochs: (int) number of epochs of training
      :batch_size: (int) batch size
      :cnn, encoder, decoder: modules of model
      :data_loader: data loader object for training
      :data_loader_eval: data loader object for evaluation
      :learning_rate: (double) learning rate
      :gradient_clip: (double) max norm of the gradients
      :n_iters: (int) number of iterations for each epoch
      :print_every: (int) number of iterations between each printing
      :use_cuda: (boolean) indicates whether the model is using a GPU
      :evaluate_with_beam_search: (boolean) indicates whether to do evaluation
       with the beam search
    Returns:
      list with loss averages
    """

    start = time.time()
    print_losses = []
    print_loss_total = 0

    # Loss and optimizer
    cnn_optimizer = torch.optim.Adam(cnn.parameters(), lr=learning_rate)
    encoder_optimizer = torch.optim.Adam(encoder.parameters(),
                                         lr=learning_rate)
    decoder_optimizer = torch.optim.Adam(decoder.parameters(),
                                         lr=learning_rate)

    criterion = nn.NLLLoss(ignore_index=0)

    for epoch in range(1, num_epochs+1):
        data_generator = data_loader.create_data_generator(args.batch_size,
                                                           args.train_path)
        data_generator2 = data_loader_eval.create_data_generator(args.batch_size_eval,
                                                                 args.validate_path)
        best_loss = None

        for iter in range(1, n_iters+1):
            (images, targets, targets_eval, num_nonzer,
             img_paths) = next(data_generator)
            loss, predicted_index, actual_index = train(images, targets,
                                                        targets_eval, cnn, encoder,
                                                        decoder, cnn_optimizer,
                                                        encoder_optimizer,
                                                        decoder_optimizer,
                                                        gradient_clip,
                                                        criterion,
                                                        args.max_length_encoder,
                                                        use_cuda)

            print_loss_total += loss

            if iter % print_every == 0:

                print_loss_avg = print_loss_total / print_every
                print_losses.append(print_loss_avg)
                print_loss_total = 0

                print('epoch #'+'%d %s (%d %d%%) %.4f' % (epoch,
                                                          timeSince(start,
                                                                    iter/float(n_iters)),
                                                          iter,
                                                          iter / float(n_iters) * 100,
                                                          print_loss_avg))
                print('Hyperparameters: batch size, lr, gradient clip '+'%d %.4f %d' % (batch_size,
                                                                                        learning_rate,
                                                                                        gradient_clip))

                print("Predicted Tokens")
                print(tokens_from_index_list(predicted_index,
                                             data_loader.tokenizer.id2vocab))
                print("Actual Tokens")
                print(tokens_from_index_list(actual_index,
                                             data_loader.tokenizer.id2vocab))

                if evaluate_with_beam_search:
                    (images, targets, targets_eval,
                     num_nonzer, img_paths) = next(data_generator2)

                    # call evaluate function with beam search
                    predicted_index, actual_index = evaluate(images, targets,
                                                             targets_eval, cnn,
                                                             encoder, decoder,
                                                             criterion,
                                                             args.max_length_encoder,
                                                             use_cuda)
                    print("Image Path")
                    print(img_paths)
                    print("Predicted Tokens")
                    print([data_loader.tokenizer.id2vocab[i]
                           for i in predicted_index])
                    print("Actual Tokens")
                    print([data_loader.tokenizer.id2vocab[i]
                           for i in actual_index])

            if not best_loss or print_loss_avg < best_loss:
                with open(args.save_cnn, 'wb') as f:
                    torch.save(cnn.state_dict(), f)
                with open(args.save_encoder, 'wb') as f:
                    torch.save(encoder.state_dict(), f)
                with open(args.save_decoder, 'wb') as f:
                    torch.save(decoder.state_dict(), f)

    return print_losses


# Create data loader
dataloader = DataLoader(args.data_base_dir, args.label_path,
                        args.max_aspect_ratio, args.max_encoder_l_h,
                        args.max_encoder_l_w, args.max_decoder_l,
                        args.max_vocab_size, token_dictionary.id2voc,
                        token_dictionary.voc2id)

dataloader_eval = DataLoader(args.data_base_dir, args.label_path,
                             args.max_aspect_ratio, args.max_encoder_l_h,
                             args.max_encoder_l_w, args.max_decoder_l,
                             args.max_vocab_size, token_dictionary.id2voc,
                             token_dictionary.voc2id)

# Create the modules of the algorithm
cnn1 = CNN()
encoder1 = EncoderBRNN(args.num_layers_encoder,
                       args.hidden_dim_encoder,
                       use_cuda)
decoder1 = AttnDecoderRNN(args.hidden_dim_encoder,
                          args.num_layers_decoder,
                          args.max_length_decoder,
                          dataloader.vocab_size,
                          args.embedding_size,
                          use_cuda)

if use_cuda:
    cnn1 = cnn1.cuda()
    encoder1 = encoder1.cuda()
    decoder1 = decoder1.cuda()
    decoder1.attn = decoder1.attn.cuda()

trainIters(args.num_epochs, args.batch_size, cnn1, encoder1, decoder1,
           dataloader, dataloader_eval, args.learning_rate, args.gradient_clip,
           n_iters=15000, print_every=1, use_cuda=use_cuda)
