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
        """
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
        """
        # Positional embeddings
        if self.use_cuda:
            hiddens = [(nn.Parameter(torch.randn(2*num_layers_encoder,
                        list_rows[0].size(1), hidden_dim_encoder // 2), requires_grad=True).cuda(),
                        nn.Parameter(torch.randn(2*num_layers_encoder,
                                                 list_rows[0].size(1), hidden_dim_encoder // 2), requires_grad=True).cuda())
                       for i in range(len(list_rows))]
        else:
            hiddens = [(nn.Parameter(torch.randn(2*num_layers_encoder,
                        list_rows[0].size(1), hidden_dim_encoder // 2), requires_grad=True),
                        nn.Parameter(torch.randn(2*num_layers_encoder,
                                                 list_rows[0].size(1), hidden_dim_encoder // 2), requires_grad=True))
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
                 embedding_size,
                 use_cuda):
        super(AttnDecoderRNN, self).__init__()
        # Keep parameters for reference
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.max_length = max_length
        self.vocab_size = vocab_size
        self.use_cuda = use_cuda
        self.embedding_size = embedding_size

        # Define layers
        self.embedding = nn.Embedding(self.vocab_size, self.embedding_size)
        self.lstm = nn.LSTM(self.embedding_size + self.hidden_size, self.hidden_size)
        self.output_context_layer = nn.Linear(2*self.hidden_size, self.hidden_size)
        self.output_layer = nn.Linear(self.hidden_size, self.vocab_size)
        self.tanh_act = nn.Tanh()

        # Create attention mechanism
        self.attn = Attn(hidden_size, max_length, use_cuda)

    def init_hidden_cell(self, batch_size, hidden_dim):
        hidden = Variable(torch.zeros(1, batch_size, hidden_dim))
        hidden = hidden.cuda() if self.use_cuda else hidden
        cell = Variable(torch.zeros(1, batch_size, hidden_dim))
        cell = cell.cuda() if self.use_cuda else cell
        return (hidden, cell)

    def init_context_output(self, batch_size, hidden_dim):
        context_output = Variable(torch.zeros(1, batch_size, hidden_dim))
        context_output = context_output.cuda() if self.use_cuda else context_output
        return context_output

    def forward(self,
                input,
                last_output_context,
                last_hidden,
                last_cell_state,
                encoder_outputs):
        """Arguments:
            input (tensor size=(batch_size, 1)): actual tokens of the last step
            hidden (tensor size=(batch_size, hidden_dim_encoder)): hidden tensor for the lstm
            cell_state (tensor size=(batch_size, hidden_dim_encoder)): cell state for the lstm
            encoder_outputs (tensor size=(max_length, batch_size, hidden_dim_encoder)): outputs of the encoder
        """
        # Note: we run this one step at a time

        # Get the embedding of the current input word (last output word)
        token_embedded = self.embedding(input)

        # Permute token_embedded to have size 1xBatch_sizexHidden_dim
        token_embedded = token_embedded.permute(1, 0, 2)

        # Combine embedded input word and last context, run through LSTM
        rnn_input = torch.cat((token_embedded, last_output_context), 2)
        rnn_output, (hidden, cell_state) = self.lstm(rnn_input,
                                                     (last_hidden, last_cell_state))

        # Calculate attention from current RNN state and all encoder outputs
        attn_weights = self.attn(rnn_output, encoder_outputs)

        # Apply attention to encoder outputs to get context
        # Permuting is made to have batch matrix multiplication with correct dimensions
        # torch.bmm(batch_sizeX1xencoder_size, batch_sizeXencoder_sizeXhidden_dim)
        context = torch.bmm(attn_weights.permute(2, 1, 0), encoder_outputs.permute(1, 0, 2))

        # Calculate context output layer
        # Permuting is made to have context with dim 1Xbatch_sizeXhidden_dim
        context_output = self.output_context_layer(torch.cat((hidden, context.permute(1, 0, 2)), 2))
        context_output = self.tanh_act(context_output)

        # Final output layer (next token prediction) using the RNN hidden state
        output = F.log_softmax(self.output_layer(context_output).squeeze(0), dim=1)
        return output, context_output, hidden, cell_state


class Attn(nn.Module):
    def __init__(self, hidden_size, max_length, use_cuda):
        super(Attn, self).__init__()
        self.hidden_size = hidden_size
        self.attn_layer = nn.Linear(self.hidden_size*2, self.hidden_size)
        # self.beta = nn.Parameter(torch.FloatTensor(hidden_size, 1))
        self.beta = nn.Parameter(torch.randn(hidden_size, 1))
        self.tanh = nn.Tanh()
        self.use_cuda = use_cuda

    def forward(self, hidden, encoder_outputs):
        seq_len = encoder_outputs.size()[0]
        batch_size = encoder_outputs.size()[1]

        # Create variable to store attention energies
        attn_energies = Variable(torch.zeros(seq_len, batch_size, 1))
        attn_energies = attn_energies.cuda() if self.use_cuda else attn_energies

        # Calculate energies for encoder outputs
        for i in range(seq_len):
            attn_energies[i] = self.score(hidden, encoder_outputs[i])

        # Normalize energies to weights in range 0 to 1
        attn_weights = F.softmax(attn_energies.permute(0, 2, 1), dim=0)
        return attn_weights

    def score(self, hidden, encoder_output):
        energy = self.attn_layer(torch.cat((hidden.squeeze(0), encoder_output), 1))
        energy = self.tanh(energy)
        energy = torch.mm(energy, self.beta)
        return energy
