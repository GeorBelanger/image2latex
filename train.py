"""
Training script

"""
import ipdb
import os
import time
import argparse
from data_gen import DataGen
import ipdb
import torch
import torch.nn as nn
from torch.autograd import Variable

#from cnn import CNN
from model import CNN
from model import EncoderBRNN
from model import AttnDecoderRNN

from utils import timeSince

use_cuda = torch.cuda.is_available()

# Create argument parser
parser = argparse.ArgumentParser()

# Data paths
parser.add_argument('--data_base_dir', type=str, default='../data/images_processed',
	help = 'Path of folder with processed images')
parser.add_argument('--label_path', type=str, default='../data/im2latex_formulas.norm.lst',
	help = 'Path of file with the latex formulas, one per line')
parser.add_argument('--data_path', type=str, default="../data/im2latex_train_filter.lst",
	help = 'Path of file with the name of the image and its corresponding line number')
parser.add_argument('--vocabulary', type=str, default='../data/latex_vocab.txt',
	help = 'Path of file with the latex vocabulary, one per line')

# Max and mins for data generator
parser.add_argument('--max_aspect_ratio', type = float, default = 10000,
	help = 'Maximum permited aspect ratio of images' )
parser.add_argument('--max_encoder_l_h', type = float, default = 10000,
	help = 'Maximum permited size for the image height')
parser.add_argument('--max_encoder_l_w', type = float, default = 10000,
	help = 'Maximum permited size for the image width')
parser.add_argument('--max_decoder_l', type = float, default = 10000,
	help = 'Maximum permited size (number of tokens) for the associated latex formula')

# Hyperparameters
parser.add_argument('--num_epochs', type = int, default = 5,
	help = 'Number of epochs for training')
parser.add_argument('--batch_size', type = int, default = 5,
	help = 'Batch size for training')
parser.add_argument('--learning_rate', type = float, default = 0.001,
	help = 'Initial learning rate for training')

# Encoder
parser.add_argument('--num_layers_encoder', type = int, default = 5,
	help = 'Number of layers in the bidirectional recurrent neural network used for encoding')
parser.add_argument('--hidden_dim_encoder', type = int, default = 512)
parser.add_argument('--max_lenth_encoder', type = int, default = 100)

# Decoder
parser.add_argument('--output_dim_decoder', type = int, default = 100)
parser.add_argument('--num_layers_decoder', type = int, default = 3)
parser.add_argument('--max_length_decoder', type = int, default = 100)

# parse arguments
args = parser.parse_args()

def train(images, targets, targets_eval, cnn, encoder, decoder, cnn_optimizer, encoder_optimizer, decoder_optimizer, criterion, max_length):
	# encoder_hidden = encoder.initHidden() # I think it works better if the hidden are initialized in the forward pass
	  
	cnn_optimizer.zero_grad()
	encoder_optimizer.zero_grad()
	decoder_optimizer.zero_grad()

	#input_length = input_variable.size()[0]
	#target_length = target_variable.size()[0]



	images = Variable(images)
	#targets = Variable(targets)
	#targets_eval = Variable(targets_eval)
	
	loss = 0

	#Forward
	#convolutional network
	list_rows = cnn(images)

	#encoder
	list_outputs, list_hiddens = encoder(list_rows) 

	#decoder
	for i in range(len(list_outputs)):
		encoder_output = list_outputs[i] # output features for every time step
		encoder_outputs = Variable(torch.zeros(max_length, encoder.batch_size, encoder.hidden_dim_encoder))
		encoder_outputs = encoder_outputs.cuda() if use_cuda else encoder_outputs

		# The encoder output should be inside the encoder_outputs of size (max_length, batch_size, hidden_dim)
		for j in range(encoder_output.size(0)):
			encoder_outputs[j] = encoder_output[j]
		encoder_outputs = encoder_outputs.permute(1, 0, 2)

		decoder_hidden = list_hiddens[i][0][-1] #do we need only the hidden or also the cell state? #hidden state of last step of encoder (thats why i use [-1]) #shape batch_size, 256)

		use_teacher_forcing = True

		if use_teacher_forcing:
			# teacher forcing: feed the target as the next input
			for di in range(len(targets)):

				
				decoder_input = targets.narrow(1,di,1) #maybe we dont need this # we want the first, second, etc. targets of each batch 
				decoder_input = torch.LongTensor(decoder_input.numpy().astype(int)) #convert to longtensor
				decoder_input = Variable(decoder_input)

				decoder(decoder_input, decoder_hidden, encoder_outputs)
				#decoder(y_onehot, decoder_hidden, encoder_output)
				#decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_output, encoder_outputs)



	"""
	#for ei in range(input_length): #figure out how do we need to do this for the cnn and encoder

	for i in range(len(list_outputs)):
		
		decoder_hidden = list_hiddens[i] #list_hidden
		#list_output	

		#figure out how to call this properly
		#decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_output, encoder_outputs)

		#loss += criterion(decoder_output, target_variable[di])

	"""

	return loss





def trainIters(batch_size, cnn, encoder, decoder, data_generator, learning_rate, n_iters, print_every = 100):
	### Work in Progress (train function needs to be complete for trainIters to work properly)
	start = time.time()
	plot_losses = []
	print_loss_total = 0
	plot_loss_total = 0

	# Loss and optimizer
	cnn_optimizer = torch.optim.Adam(cnn.parameters(), lr = learning_rate)
	encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr = learning_rate)
	decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr = learning_rate)

	criterion = nn.NLLLoss()

	data_generator.shuffle_lines()	
	
	for iter in range(1, n_iters+1):
		images, targets, targets_eval, num_nonzeros, img_paths = data_generator.next_batch(args.batch_size)

		loss = train(images, targets, targets_eval, cnn, encoder, decoder, cnn_optimizer, encoder_optimizer, decoder_optimizer, criterion, args.max_lenth_encoder)

		print_loss_total += loss
		plot_loss_total += loss

		if iter % print_every == 0:
			print_loss_avg = print_loss_total / print_every
			print_loss_total = 0
			print('%s (%d %d%%) %.4f' % (timeSince(start, iter/n_iters),
				iter, iter / n_iters * 100, print_loss_avg))

		if iter % plot_every == 0:
			plot_loss_avg = plot_loss_total / plot_every
			plot_losses.append(plot_loss_avg)
			plot_loss_total = 0
	
	return plot_losses
"""
if using a notebook, we can print the results. (showPlot is missing)
	showPlot(plot_losses)
"""
	

# Create data generator
datagen = DataGen(args.data_base_dir, args.data_path, args.label_path, args.max_aspect_ratio, 
	args.max_encoder_l_h, args.max_encoder_l_w, args.max_decoder_l)

# Create the modules of the algorithm
cnn1 = CNN()
encoder1 = EncoderBRNN(args.batch_size, args.num_layers_encoder, args.hidden_dim_encoder)
decoder1 = AttnDecoderRNN(args.hidden_dim_encoder//2, args.output_dim_decoder, args.num_layers_decoder, args.max_length_decoder, datagen.vocab_size)


if use_cuda:
    cnn1 = ccn1.cuda()
    encoder1 = encoder1.cuda()
    attn_decoder1 = attn_decoder1.cuda()

ipdb.set_trace()

trainIters(args.batch_size, cnn1, encoder1, decoder1, datagen, args.learning_rate, n_iters=75000, print_every=5000)

"""

# Loss and optimizer #(trainIters)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(cnn.parameters(), lr = args.learning_rate)
optimizer = torch.optim.Adam(encoder.parameters(), lr = args.learning_rate)
#optimizer = torch.optim.Adam(cnnrnn.parameters(), lr = args.learning_rate)

# Train the model
for epoch in range(args.num_epochs): # Still missing epochs
	
	# Call data generator (outside)
	datagen = DataGen(args.data_base_dir, args.data_path, args.label_path, args.max_aspect_ratio, 
	args.max_encoder_l_h, args.max_encoder_l_w, args.max_decoder_l)

	# Shuffle data #(TrainIters)
	datagen.shuffle()


	for i in range(10):
		
		# Generate a batch (#train_iters)
		images, targets, targets_eval, num_nonzeros, img_paths = datagen.next_batch(args.batch_size)

		# Wrap tensor in Variable object (#train)
		images = Variable(images)
		targets = Variable(targets)
		targets_eval = Variable(targets_eval)

		# Forward 
		optimizer.zero_grad()
		ipdb.set_trace()
		
		list_rows = cnn(images)
		list_outputs, list_hiddens = encoder(list_rows)


		for i in len(list_outputs):
			
			decoder_hidden = list_hiddens[i]
			list_output		


		# Backward + Optimize (Work in progress)
		# loss = criterion(outputs, labels)
		# loss.backward()
		# optimizer.step()
"""	





	
	




