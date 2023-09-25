import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

from weigth_sage import WeightedSAGEConv

parser = argparse.ArgumentParser()
parser.add_argument('--device',type=str,default='cpu',help='')
parser.add_argument('--data',type=str,default='data/METR-LA',help='data path')
parser.add_argument('--lr',type=float,default=0.01,help='learning rate')
parser.add_argument('--adj_data', type=str,default='data/sensor_graph/adj_mx.pkl',help='adj data path')
parser.add_argument('--horizon', type=int, default=4)
parser.add_argument('--prediction_window', type=int, default=4)
parser.add_argument('--step_size',type=int,default=100,help='step_size')
parser.add_argument('--save', type=str, default='model/model.pt',
                    help='path to save the final model')
parser.add_argument('--gcn_depth',type=int,default=1,help='graph convolution depth')
parser.add_argument('--num_nodes',type=int,default=49,help='number of nodes/variables')
parser.add_argument('--dropout',type=float,default=0.2,help='dropout rate')
parser.add_argument('--subgraph_size',type=int,default=20,help='k')
parser.add_argument('--node_dim',type=int,default=16,help='dim of nodes')
parser.add_argument('--init_embedding_dim',type=int,default=49,help='dim of embedding')
parser.add_argument('--dilation_exponential',type=int,default=1,help='dilation exponential')
parser.add_argument('--optim', type=str, default='adam')
parser.add_argument('--conv_channels',type=int,default=4,help='convolution channels')
parser.add_argument('--residual_channels',type=int,default=4,help='residual channels')
parser.add_argument('--skip_channels',type=int,default=8,help='skip channels')
parser.add_argument('--end_channels',type=int,default=16,help='end channels')
parser.add_argument('--in_dim',type=int,default=1,help='inputs dimension')
parser.add_argument('--seq_in_len',type=int,default=12,help='input sequence length')
parser.add_argument('--seq_out_len',type=int,default=1,help='output sequence length')
parser.add_argument('--layers',type=int,default=3,help='number of layers')
parser.add_argument('--batch_size',type=int,default=100,help='batch size')
parser.add_argument('--learning_rate',type=float,default=0.005,help='learning rate')
parser.add_argument('--weight_decay',type=float,default=0.0001,help='weight decay rate')
parser.add_argument('--clip',type=int,default=5,help='clip')
parser.add_argument('--step_size1',type=int,default=2500,help='step_size')
parser.add_argument('--step_size2',type=int,default=100,help='step_size')
parser.add_argument('--L1Loss', type=bool, default=True)
parser.add_argument('--epochs',type=int,default=200,help='')
parser.add_argument('--print_every',type=int,default=50,help='')
parser.add_argument('--seed',type=int,default=101,help='random seed')
parser.add_argument('--expid',type=int,default=1,help='experiment id')
parser.add_argument('--propalpha',type=float,default=0.05,help='prop alpha')
parser.add_argument('--tanhalpha',type=float,default=1,help='adj alpha')
parser.add_argument('--num_split',type=int,default=1,help='number of splits for graphs')
parser.add_argument('--normalize', type=int, default=0)
parser.add_argument('--runs',type=int,default=10,help='number of runs')
parser.add_argument('--autoregressive_retrain', type = bool, default=True, help='autoregressive training or not')
parser.add_argument('--multivariate_TE_enhanced', type = bool, default=True, help='Multivariate Transfer Entropy')
parser.add_argument('--pure_data_driven', type = bool, default=True, help='Use purely data driven for adaptive learning')
parser.add_argument('--BivariateTE', type = bool, default=True, help='Using Bivaraite Transfer Entropy')
parser.add_argument('--Plot_4_smaller_regions', type = bool, default=False, help='Plot for smaller regions')
parser.add_argument('--Vector_Autoregression', type = bool, default=False, help='Perform VAR baseline')
parser.add_argument('--train_horizon',  type=int, default=1, help = 'specify the horizon for training stage')

args = parser.parse_args()


# Assuming data is a numpy array of shape (119, 51)
def create_dataset(data, input_length, output_length):
    X, y = [], []
    for i in range(len(data) - input_length - output_length + 1):
        X.append(data[i:i + input_length, :])
        y.append(data[i + input_length:i + input_length + output_length, :])
    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

# Define the model
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h_n, _ = self.lstm(x)
        out = self.fc(h_n[:, -1, :])
        return out.view(x.size(0), 4, 49)

class GRU():


def LSTM():
    length = hopsitalization.shape[0]

    input_length = 12
    output_length = 4

    train_idx = int(length * 0.8)
    val_idx = length - 4
    # Split data into training, validation, and test sets
    train_data = hopsitalization[:train_idx, :]
    val_data = hopsitalization[train_idx:val_idx, :]
    test_data = hopsitalization[val_idx:, :]

    X_train, y_train = create_dataset(train_data, input_length, output_length)
    X_val, y_val = create_dataset(val_data, input_length, output_length)
    X_test, y_test = create_dataset(test_data, input_length, output_length)

    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    test_dataset = TensorDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)
    test_loader = DataLoader(test_dataset, batch_size=32)

    input_dim = 49
    hidden_dim = 8
    output_dim = 4 * 49

    model = LSTMModel(input_dim, hidden_dim, output_dim)

    # Define loss and optimizer
    loss_function = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    epochs = 100
    training_loss = list()
    val_losses = list()
    for epoch in range(epochs):
        # Training
        model.train()
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            predictions = model(X_batch)
            loss = loss_function(predictions, y_batch)
            loss.backward()
            optimizer.step()
        # Validation
        model.eval()
        with torch.no_grad():
            val_loss = 0
            for X_batch, y_batch in val_loader:
                predictions = model(X_batch)
                val_loss += loss_function(predictions, y_batch).item()
            val_loss /= len(val_loader)
            print(f"Epoch {epoch + 1}/{epochs}, Training Loss: {loss} Validation Loss: {val_loss}")
        training_loss.append(loss.detach().numpy())
        val_losses.append(val_loss)
    plt.plot(training_loss, color='blue')
    plt.plot(val_losses, color='red')
    plt.show()
    # Testing
    model.eval()
    with torch.no_grad():
        test_loss = 0
        for X_batch, y_batch in test_loader:
            predictions = model(X_batch)
            test_loss += loss_function(predictions, y_batch).item()
        test_loss /= len(test_loader)
        print(f"Test Loss: {test_loss}")

def DCRNN():


def VAR():





if __name__ == '__main__':
    hopsitalization = pd.read_csv('hosp_weekly_filt_case_data.csv')
    hopsitalization = hopsitalization.iloc[:-1, :]
    hopsitalization = hopsitalization.reset_index()
    hopsitalization['state'] = hopsitalization['state'].astype(int)

    fips_list = hopsitalization['state'].values
    fips_2_index = hopsitalization.set_index('state')['index'].to_dict()
    index_2_fips = hopsitalization.set_index('index')['state'].to_dict()

    neighbouring_states = pd.read_csv('neighbouring_states.csv')
    fips_states = pd.read_csv('state_hhs_map.csv', header=None).iloc[:(-3), :]
    fips_2_state = fips_states.set_index(0)[2].to_dict()
    state_2_fips = fips_states.set_index(2)[0].to_dict()
    state_len = len(fips_list)
    hopsitalization['state'] = hopsitalization['state'].map(fips_2_state)

    index_2_state = hopsitalization.set_index('index')['state'].to_dict()
    state_2_index = hopsitalization.set_index('state')['index'].to_dict()

    neighbouring_states['StateCode'] = neighbouring_states['StateCode'].map(state_2_index)
    neighbouring_states['NeighborStateCode'] = neighbouring_states['NeighborStateCode'].map(state_2_index)

    G = nx.from_pandas_edgelist(neighbouring_states, 'StateCode', 'NeighborStateCode')

    hopsitalization = hopsitalization.iloc[:, 30:]  # remove all 0 datapoints
    hopsitalization = hopsitalization.T.values
    models = ['LSTM','GRU','DCRNN','VAR']
    for model in models:
        if model == 'LSTM':
            LSTM()
        elif model == 'GRU':
            GRU()
        elif model == 'DCRNN':
            DCRNN()
        else:
            VAR()



