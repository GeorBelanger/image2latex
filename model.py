import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import ipdb


class CNN(nn.Module):
    """Create the CNN model:
    Layer 1: CONV2D -> RELU ->  MAXPOOL ->
    Layer 2: CONV2D -> RELU -> MAXPOOL ->
    Layer 3: CONV2D -> BATCHNORM -> RELU ->
    Layer 4: CONV2D -> RELU -> MAXPOOL ->
    Layer 5: CONV2D -> BATCHNORM -> RELU -> MAXPOOL ->
    Layer 6: CONV2D -> BATCHNORM -> RELU -> """
    def __init__(self):
        super(CNN, self).__init__()
        self.layer1CNN = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3,
                      stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2CNN = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3,
                      stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer3CNN = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3,
                      stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU())
        self.layer4CNN = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3,
                      stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1), padding=0))
        self.layer5CNN = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3,
                      stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2), padding=0))
        self.layer6CNN = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3,
                      stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU())

    def forward(self, x):
        """Make the forward pass for the CNN
        Arguments:
        x -- input data of shape (batch_size, 1, imgH, imgW)
        Returns:
        list of #H rows of the image features after
        passing throught the CNN
        Each row has shape (W, batch_size, 512)"""
        out = self.layer1CNN(x)
        out = self.layer2CNN(out)
        out = self.layer3CNN(out)
        out = self.layer4CNN(out)
        out = self.layer5CNN(out)
        out = self.layer6CNN(out)
        out = out.permute(2, 3, 0, 1)
        list_rows = torch.unbind(out, dim=0)

        return list_rows


class EncoderBRNN(nn.Module):
    """Create a bidirectional recurrent neural network
    (using Long-Short Term Memory cells) to encode the
    rows of the features that the convolutional network generated
    Arguments:
        num_layers_encoder (int): number of layers of the encoder
        hidden_dim_encoder (int): hidden dimension for the encoder
        use_cuda (boolean): indicates whether the model is using a GPU

    """
    def __init__(self, num_layers_encoder, hidden_dim_encoder,
                 use_cuda):
        super(EncoderBRNN, self).__init__()
        self.hidden_dim_encoder = hidden_dim_encoder
        self.num_layers_encoder = num_layers_encoder
        self.brnn = nn.LSTM(512, hidden_dim_encoder // 2,
                            num_layers_encoder, bidirectional=True)
        self.use_cuda = use_cuda

    def init_hidden_cell(self, list_rows, num_layers_encoder,
                         hidden_dim_encoder):
        # Initialize hidden states
        # The number of layers used is 2*num_layers_encoder
        # because its a bidirectional RNN
        # enable Variables of hidden state to be used by CUDA
        if self.use_cuda:
            hiddens = [(Variable(torch.zeros(2*num_layers_encoder,
                       list_rows[0].size(1), hidden_dim_encoder // 2)).cuda(),
                       Variable(torch.zeros(2*num_layers_encoder,
                                            list_rows[0].size(1),
                                            hidden_dim_encoder // 2)).cuda())
                       for i in range(len(list_rows))]
        else:
            hiddens = [(Variable(torch.zeros(2*num_layers_encoder,
                       list_rows[0].size(1), hidden_dim_encoder // 2)),
                       Variable(torch.zeros(2*num_layers_encoder,
                                            list_rows[0].size(1),
                                            hidden_dim_encoder // 2)))
                       for i in range(len(list_rows))]
        return hiddens

    def forward(self, list_rows):
        """Make the forward pass for the Bidirectional RNN
        Arguments:
            list_rows -- list of H rows of the image features after
            passing through the CNN
            Each row has shape (W, batch_size, 512)
            (H and W are the integral part of the original height
            and width of the image divided by 8)
        Returns:
            list of H tensor of shape (W, batch_size, 512)
            Each tensor is the output that the Bidirectional RNN generated for
            each row in list_rows for each point in time t = 1, 2, ..., W
            (H and W are the integral part of the original height and
            width of the image divided by 8)"""

        hiddens = self.init_hidden_cell(list_rows, self.num_layers_encoder,
                                        self.hidden_dim_encoder)

        # List for outputs of encoder
        list_outputs = []
        list_hiddens = []

        # Pass each row through the brnn
        for i, row in enumerate(list_rows):
            output, hidden = self.brnn(row, hiddens[i])

            list_outputs.append(output)
            list_hiddens.append(hidden)

        outputs = torch.cat(list_outputs, 0)

        return outputs


class AttnDecoderRNN(nn.Module):
    """Create the recurrent neural network decoder with an attention mechanism
    Arguments:
        hidden_size (int): hidden size of the decoder
        output_size (int): output size of the decoder
        n_layers (int): number of layers of the decoder
        max_length: maximum length of encoder_outputs
        vocab_size: size of vocabulary of tokens
    """

    def __init__(self,
                 hidden_size,
                 output_size,
                 n_layers,
                 max_length,
                 vocab_size,
                 use_cuda):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.max_length = max_length
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(self.vocab_size, self.output_size)
        self.attn = nn.Linear(1512, self.max_length)
        self.attn_combine = nn.Linear(1512, 2 * self.hidden_size)
        self.lstm = nn.LSTM(2*self.hidden_size, 2*self.hidden_size)
        self.out = nn.Linear(2*self.hidden_size, self.vocab_size)
        self.use_cuda = use_cuda
        # self.last_hidden = torch.zeros(hidden_size)
        # self.last_cell = torch.zeros(hidden_size)

    def init_hidden_cell(self, batch_size, hidden_dim):
        hidden = Variable(torch.zeros(batch_size, hidden_dim))
        hidden = hidden.cuda() if self.use_cuda else hidden
        cell = Variable(torch.zeros(batch_size, hidden_dim))
        cell = cell.cuda() if self.use_cuda else cell
        return (hidden, cell)

    def forward(self, input, encoder_outputs, hidden=None, cell_state=None, init_hidden=False):
        """Make the forward pass for the decoder network
        Arguments:
            input (tensor size=(batch_size, 1)): actual tokens of the last step
            hidden (tensor size=(batch_size, hidden_dim_encoder)): hidden tensor for the lstm
            cell_state (tensor size=(batch_size, hidden_dim_encoder)): cell state for the lstm
            encoder_outputs (tensor size=(max_length, batch_size, hidden_dim_encoder)): outputs of the encoder
        """
        if hidden is None:
            hidden = self.last_hidden
        if cell_state is None:
            cell_state = self.last_cell
        if init_hidden:
            hidden, cell = self.init_hidden_cell()

        # create attention weights and apply it to output
        embedded = self.embedding(input).squeeze(1)
        attn_weights = F.softmax(self.attn(torch.cat((embedded, hidden), 1)))
        attn_applied = torch.bmm(attn_weights.unsqueeze(1),
                                 encoder_outputs.permute(1, 0, 2)).squeeze(1)
        output = torch.cat((embedded, attn_applied), 1)
        output = self.attn_combine(output)

        output = output.unsqueeze(0)
        cell_state = cell_state.unsqueeze(0)
        hidden = hidden.unsqueeze(0)
        self.last_cell = cell_state
        self.last_hidden = hidden

        # pass through the LSTM
        for i in range(self.n_layers):
            output = F.relu(output)
            output, (hidden, cell_state) = self.lstm(output,
                                                     (hidden, cell_state))
        output = F.log_softmax(self.out(output[0]))

        return output, hidden.squeeze(0), cell_state.squeeze(0)
