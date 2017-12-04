import torch
import torch.nn as nn


class CNN(nn.Module):
	"""
	Creates the CNN model:
	Layer 1: CONV2D -> RELU ->  MAXPOOL -> 
	Layer 2: CONV2D -> RELU -> MAXPOOL -> 
	Layer 3: CONV2D -> BATCHNORM -> RELU -> 
	Layer 4: CONV2D -> RELU -> MAXPOOL -> 
	Layer 5: CONV2D -> BATCHNORM -> RELU -> MAXPOOL -> 
	Layer 6: CONV2D -> BATCHNORM -> RELU -> 
	Layer 7: TRANSPOSE -> SPLIT_BY_LINES


	torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0)
	"""
	def __init__(self):
		super(CNN, self).__init__()
		self.layer1 = nn.Sequential(
			nn.Conv2d(in_channels = 1, out_channels = 64, kernel_size = 3, stride = 1, padding = 1), #tensor.shape=(batch_size, 64, imgH, imgW)
			nn.ReLU(),
			nn.MaxPool2d(kernel_size = 2, stride = 2)) #tensor.shape=batch_size, 64, imgH/2, imgW/2)
		self.layer2 = nn.Sequential(
			nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size = 3, stride = 1, padding = 1), #tensor.shape=(batch_size, 128, imgH/2, imgW/2)
			nn.ReLU(),
			nn.MaxPool2d(kernel_size = 2, stride = 2)) #tensor.shape=(batch_size, 128, imgH/2/2, imgW/2/2)
		self.layer3 = nn.Sequential(
			nn.Conv2d(in_channels = 128, out_channels = 256, kernel_size = 3, stride = 1, padding = 1), #tensor.shape=(batch_size, 256, imgH/2/2, imgW/2/2)
			nn.BatchNorm2d(256),
			nn.ReLU())
		self.layer4 = nn.Sequential(
			nn.Conv2d(in_channels = 256, out_channels = 256, kernel_size = 3, stride = 1, padding = 1), #tensor.shape=(batch_size, 256, imgH/2/2, imgW/2/2)
			nn.ReLU(),
			nn.MaxPool2d(kernel_size = (2,1), stride = (2,1), padding = 0)) #kH = 2, kW = 1, #tensor.shape=(batch_size, 256, imgH/2/2/2, imgW/2/2)
		self.layer5 = nn.Sequential(
			nn.Conv2d(in_channels = 256, out_channels = 512, kernel_size = 3, stride = 1, padding = 1), #tensor.shape=(batch_size, 512, imgH/2/2/2, imgW/2/2)
			nn.BatchNorm2d(512),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size = (1,2), stride = (1,2), padding = 0)) #kH = 1, kW = 2, #tensor.shape=(batch_size, 512, imgH/2/2/2, imgW/2/2/2)
		self.layer6 = nn.Sequential(
			nn.Conv2d(in_channels = 512, out_channels= 512, kernel_size = 3, stride = 1, padding = 1), #tensor.shape=(batch_size, 512, imgH/2/2/2, imgW/2/2/2)
			nn.BatchNorm2d(512),
			nn.ReLU())
		#self.layer7 = nn.Sequential( # If we define H = imgH/8 and W = imgW/8, then tensor.shape= (batch_size, 512, H, W)
			#nn.Transpose({2,3}, {3,4}) # tensor.shape = (batch_size, H, W, 512)
			#nn.SplitTable(1,3)) #tensor.shape = H lists of (batch_size, W, 512) ?


	def forward(self, x):
		out = self.layer1(x)
		out = self.layer2(out)
		out = self.layer3(out)
		out = self.layer4(out)
		out = self.layer5(out)
		out = self.layer6(out)
		out = out.permute(2,0,1,3)
		list_rows = torch.unbind(out, dim = 0)
		#out = torch.chunk(outputs, outputs.size()[0], dim = 0)
		# torch.squeeze?
		#out = out.view(out.size(0), -1)
		#out = self.layer7(out)
		return out
