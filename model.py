import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import ipdb


class CNN(nn.Module):
	"""
	Creates the CNN model:
	Layer 1: CONV2D -> RELU ->  MAXPOOL -> 
	Layer 2: CONV2D -> RELU -> MAXPOOL -> 
	Layer 3: CONV2D -> BATCHNORM -> RELU -> 
	Layer 4: CONV2D -> RELU -> MAXPOOL -> 
	Layer 5: CONV2D -> BATCHNORM -> RELU -> MAXPOOL -> 
	Layer 6: CONV2D -> BATCHNORM -> RELU -> 
	
	"""
	def __init__(self):
		super(CNN, self).__init__()

		self.layer1CNN = nn.Sequential(
			nn.Conv2d(in_channels = 1, out_channels = 64, kernel_size = 3, stride = 1, padding = 1), #tensor.shape=(batch_size, 64, imgH, imgW)
			nn.ReLU(),
			nn.MaxPool2d(kernel_size = 2, stride = 2)) #tensor.shape=batch_size, 64, imgH/2, imgW/2)
		self.layer2CNN = nn.Sequential(
			nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size = 3, stride = 1, padding = 1), #tensor.shape=(batch_size, 128, imgH/2, imgW/2)
			nn.ReLU(),
			nn.MaxPool2d(kernel_size = 2, stride = 2)) #tensor.shape=(batch_size, 128, imgH/2/2, imgW/2/2)
		self.layer3CNN = nn.Sequential(
			nn.Conv2d(in_channels = 128, out_channels = 256, kernel_size = 3, stride = 1, padding = 1), #tensor.shape=(batch_size, 256, imgH/2/2, imgW/2/2)
			nn.BatchNorm2d(256),
			nn.ReLU())
		self.layer4CNN = nn.Sequential(
			nn.Conv2d(in_channels = 256, out_channels = 256, kernel_size = 3, stride = 1, padding = 1), #tensor.shape=(batch_size, 256, imgH/2/2, imgW/2/2)
			nn.ReLU(),
			nn.MaxPool2d(kernel_size = (2,1), stride = (2,1), padding = 0)) #kH = 2, kW = 1, #tensor.shape=(batch_size, 256, imgH/2/2/2, imgW/2/2)
		self.layer5CNN = nn.Sequential(
			nn.Conv2d(in_channels = 256, out_channels = 512, kernel_size = 3, stride = 1, padding = 1), #tensor.shape=(batch_size, 512, imgH/2/2/2, imgW/2/2)
			nn.BatchNorm2d(512),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size = (1,2), stride = (1,2), padding = 0)) #kH = 1, kW = 2, #tensor.shape=(batch_size, 512, imgH/2/2/2, imgW/2/2/2)
		self.layer6CNN = nn.Sequential(
			nn.Conv2d(in_channels = 512, out_channels= 512, kernel_size = 3, stride = 1, padding = 1), #tensor.shape=(batch_size, 512, imgH/2/2/2, imgW/2/2/2)
			nn.BatchNorm2d(512),
			nn.ReLU())

	def forward(self, x):
		"""
		Makes the forward pass for the CNN

		Arguments:
		x -- input data of shape (batch_size, 1, imgH, imgW)

		Returns:
		list_rows -- list of #H rows of the image features after passing throught the CNN 
		Each row has shape (W, batch_size, 512)
		"""
		out = self.layer1CNN(x)  #tensor.shape=batch_size, 64, imgH/2, imgW/2)
		out = self.layer2CNN(out) #tensor.shape=(batch_size, 128, imgH/2/2, imgW/2/2)
		out = self.layer3CNN(out) #tensor.shape=(batch_size, 256, imgH/2/2, imgW/2/2)
		out = self.layer4CNN(out) #tensor.shape=(batch_size, 256, imgH/2/2/2, imgW/2/2)
		out = self.layer5CNN(out) #tensor.shape=(batch_size, 512, imgH/2/2/2, imgW/2/2/2)
		out = self.layer6CNN(out) #tensor.shape=(batch_size, 512, imgH/2/2/2, imgW/2/2/2) 
		out = out.permute(2,3,0,1) #If we define H = imgH/8 and W = imgW/8, then tensor.shape = (H, W, batch_size, 512)
		list_rows = torch.unbind(out, dim = 0) # tuple of size H contaning tensors of shape (W, batch_size, 512)

		return list_rows

class EncoderBRNN(nn.Module):
	"""
	Create a bidirectional recurrent neural network (using Long-Short Term Memory cells) 
	to encode the rows of the features that the convolutional network generated

	"""
	def __init__(self, batch_size, num_layers_encoder, hidden_dim_encoder):
		super(EncoderBRNN, self).__init__()
		self.hidden_dim_encoder = 512
		self.num_layers_encoder = num_layers_encoder
		self.batch_size = batch_size

		self.brnn = nn.LSTM(512, hidden_dim_encoder // 2, num_layers_encoder, bidirectional = True)

	def forward(self, list_rows):
		"""
		Make the forward pass for the Bidirectional RNN

		Arguments:
		list_rows -- list of H rows of the image features after passing through the CNN 
		Each row has shape (W, batch_size, 512)
		(H and W are the integral part of the original height and width of the image divided by 8)
	
		Returns:
		list_output -- list of H tensor of shape (W, batch_size, 512) 
		Each tensor is the output that the Bidirectional RNN generated for each row in list_rows for each point in time t = 1, 2, ..., W
		(H and W are the integral part of the original height and width of the image divided by 8)

		"""

		# Initialize hidden states (this needs to be changed so its also learned by the model)
		# From paper: in order to capture the sequential order information in vertical direction, 
		# we use a trainable hidden state V_{h,0}, for each row, which we refer to as positional embedding
		# The num_layers is 2*nlayers defined in the self.brnn because its bidirectional
		hiddens = [(Variable(torch.zeros(2*self.num_layers_encoder, self.batch_size, self.hidden_dim_encoder // 2)), Variable(torch.zeros(2*self.num_layers_encoder, self.batch_size, self.hidden_dim_encoder // 2))) 
		for i in range(len(list_rows))]

		# List for outputs of encoder
		list_outputs = []
		list_hiddens = []

		# Pass each row through the brnn
		for i, row in enumerate(list_rows):
			output, hidden = self.brnn(row, hiddens[i])
			
			list_outputs.append(output) #take the last hidden state of the rnn (if backprop doesnt work with append, use torch.cat)
			list_hiddens.append(hidden)

		return list_outputs, list_hiddens

	def initHidden(self):
		result = [(Variable(torch.zeros(2*self.num_layers_encoder, self.batch_size, self.hidden_dim_encoder // 2)), Variable(torch.zeros(2*self.num_layers_encoder, self.batch_size, self.hidden_dim_encoder // 2))) 
		for i in range(len(list_rows))]

		if use_cuda:
			return result.cuda()
		else:
			return result


class AttnDecoderRNN(nn.Module):
	"""
	Creates the recurrent neural network decoder with attention

	"""

	def __init__(self, hidden_size, output_size, n_layers, max_length, vocab_size):
		super(AttnDecoderRNN, self).__init__()
		self.hidden_size = hidden_size
		self.output_size = output_size
		self.n_layers = n_layers
		self.max_length = max_length
		self.vocab_size = vocab_size
		self.embedding = nn.Embedding(self.vocab_size, self.output_size)

		#self.attn = nn.Linear(self.hidden_size, self.max_length)
		self.attn = nn.Linear(356, self.max_length)
		self.attn_combine = nn.Linear(612, self.hidden_size)
		#self.lstm = nn.LSTM(input_size, hidden_size, num_layers)
		#self.lstm = nn.LSTM(self.hidden_size, self.hidden_size, self.n_layers)
		self.lstm = nn.LSTM(self.hidden_size, self.hidden_size)
		self.out = nn.Linear(self.hidden_size, self.output_size)

	def forward(self, input, hidden, cell_state, encoder_outputs):

		embedded = self.embedding(input)

		attn_weights = F.softmax(self.attn(torch.cat((torch.squeeze(embedded),hidden),1))) 

		attn_applied = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs)

		output = torch.cat((embedded.squeeze(), attn_applied.squeeze()),1)
		output = self.attn_combine(output)

		output = output.unsqueeze(0)
		#output = output.permute(1, 0, 2)
		cell_state = cell_state.unsqueeze(0)
		hidden = hidden.unsqueeze(0)

		ipdb.set_trace()
		for i in range(self.n_layers):
			output = F.relu(output)
			#output = F.relu(output.unsqueeze(1))
			#maybe the problem is that we only have hidden but we are forgetting the cell state
			#also try to unsqueeze hidden
			#output, hidden = self.lstm(output, hidden)
			
			output, (hidden, cell_state) = self.lstm(output, (hidden, cell_state))
		output = F.log_softmax(self.out(output[0]))
		return output
