import torch
from torch.nn import Parameter, LSTM
from weight_sage import WeightedSAGEConv
from torch_geometric.nn.inits import glorot, zeros
import torch.nn.functional as F
from torch.nn import LSTMCell, GRUCell, RNNCell, LSTM as TorchLSTM
from graph_nets import GraphLinear

from torch_geometric_temporal.nn import DCRNN, GConvLSTM, GConvGRU
from torch.nn.init import xavier_uniform

class RNN(torch.nn.Module):
    """
    Base class for Recurrent Neural Networks (LSTM or GRU).
    Initialization to this class contains all variables for variation of the model.
    Consists of one of the above RNN architectures followed by an optional GNN on the final hidden state.
    Parameters:
        node_features: int - number of features per node
        output: int - length of the output vector on each node, in our case, it's 4-dimensional output corresponding to 4 weeks ahead.
        dim: int - number of features of embedding for each node
        module: torch.nn.Module - to be used in the LSTM to calculate each gate
    """
    def __init__(self, node_features=1, output=4, dim=32, module=GraphLinear, rnn = TorchLSTM, gnn=WeightedSAGEConv, gnn_2=WeightedSAGEConv, rnn_depth=1, name="RNN", skip_connection=True):
        super(RNN, self).__init__()
        self.dim = dim
        self.rnn_depth = rnn_depth
        self.name = name
        self.skip_connection = skip_connection

        # Ensure that matrix multiplication sizes match up based on whether GNNs and RNN are used
        print(gnn)
        if gnn:
            if skip_connection:
                self.gnn = gnn(node_features, dim)
            else:
                self.gnn = gnn(node_features, dim * 2)
            if rnn:
                if skip_connection:
                    self.recurrent = rnn(dim, dim)
                else:
                    self.recurrent = rnn(dim * 2, dim * 2)
            else:
                self.recurrent = None
        else:
            self.gnn = None
            if rnn:
                self.recurrent = rnn(node_features, dim)
            else:
                self.recurrent = None
        if gnn_2:
            if gnn:
                self.gnn_2 = gnn_2(dim * 2, dim * 2)
            else:
                self.gnn_2 = gnn_2(dim + node_features, dim * 2)
        else:
            self.gnn_2 = None

        self.lin1 = torch.nn.Linear(4 * dim, dim)
        self.lin2 = torch.nn.Linear(dim, output)
        self.act1 = torch.nn.ReLU()
        self.act2 = torch.nn.ReLU()

    def forward(self, data, h=None, c=None, state_index=None):
        # Get data from snapshot
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        # First GNN Layer
        if self.gnn:
            x = self.gnn(x, edge_index, edge_attr)
            x = F.relu(x)

        state_representation = x[state_index]
        state_representation = torch.unsqueeze(state_representation,0)
        state_representation = torch.unsqueeze(state_representation,0) # batch * length * feature_length

         # batch * length * input dimensions
        # Initialize hidden and cell states if None
        current_dim = self.dim
        if not self.skip_connection:
            current_dim = self.dim * 2

        # RNN Layer
        if h is None:
            h = torch.zeros((1, 1, self.dim * 2))
        if c is None:
            c = torch.zeros((1, 1, self.dim * 2))

        if self.recurrent:
            for i in range(self.rnn_depth):
               output, (h, c) = self.recurrent(state_representation, (h, c))

        # Skip connection from first GNN
        '''
        if self.skip_connection:
            x = torch.cat((x, h), 1)
        else:
            x = h
        '''

        x = torch.cat((state_representation,h), 2)

        # Second GNN Layer
        if self.gnn_2:
            x = self.gnn_2(x, edge_index, edge_attr)

        # Readout and activation layers
        x = torch.squeeze(x, 1)
        x = self.lin1(x)
        # x = self.act1(x)
        x = self.lin2(x)
        # x = self.act2(x)

        return x, h, c
