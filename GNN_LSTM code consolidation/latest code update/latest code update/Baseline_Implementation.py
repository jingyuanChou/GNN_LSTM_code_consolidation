import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.preprocessing import MinMaxScaler
from torch_geometric_temporal import DCRNN

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

def generate_sequences(data, input_seq_len, pred_seq_len):
    input_sequences = []
    target_sequences = []
    for t in range(data.shape[1] - input_seq_len - pred_seq_len + 1):
        input_seq = data[:, t:t+input_seq_len, :]
        target_seq = data[:, t+input_seq_len:t+input_seq_len+pred_seq_len, 0]
        input_sequences.append(input_seq)
        target_sequences.append(target_seq)
    return torch.stack(input_sequences), torch.stack(target_sequences)


# Define the model
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
    def forward(self, x):
        h_n, _ = self.lstm(x)
        out = self.fc(h_n[:, -1, :])
        return out.view(x.size(0), 4, -1)

class GRUModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GRUModel, self).__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h_n, _ = self.gru(x)
        out = self.fc(h_n[:, -1, :])
        return out.view(x.size(0), 4, -1)


class TrafficPredictor(nn.Module):
    def __init__(self, in_channels, out_channels, K):
        super(TrafficPredictor, self).__init__()
        self.dcrnn = DCRNN(in_channels, out_channels, K)
        self.linear = nn.Linear(out_channels, 1)

    def forward(self, x_sequence, edge_index, edge_weight_sequence, h=None):
        predictions = []

        # Process the input sequence
        for t in range(12):
            x_t = x_sequence[:, t, :]
            edge_weight_t = edge_weight_sequence[t]
            h = self.dcrnn(x_t, edge_index, edge_weight_t, h)

        # Predict for the next timestamps
        for _ in range(4):
            h = self.dcrnn(h, edge_index, edge_weight_t, h)  # using the last edge_weight_t for simplicity
            out = self.linear(h)
            predictions.append(out.unsqueeze(1))

        return torch.cat(predictions, dim=1)

def Vector_Autoregression(data, past, horizon):
    # VAR
    # Assuming 'data' is a pandas DataFrame with your 51 variables and 119 timestamps
    data_x = data[:-4,:]
    data_y = data[-4:,:]
    model = VAR(data_x)
    fit_model = model.fit(maxlags=past)  # Choose an appropriate lag
    forecast = fit_model.forecast(data[-fit_model.k_ar:], steps=horizon)

    forecast = forecast.T
    gt = data_y.T

    list_mae_VAR = list()
    list_of_mape_VAR = list()
    for i in range(forecast.shape[0]):
        list_mae_VAR.append(sum(abs(forecast[i] - gt[i])))
        sum_mape = 0
        for j, ele in enumerate(forecast[i]):
            temp_mape = abs((ele - gt[i][j]) / gt[i][j])
            sum_mape = sum_mape + temp_mape
        list_of_mape_VAR.append(sum_mape / 4)
    mean_mae_VAR = np.mean(list_mae_VAR)
    gt = np.sum(gt, axis=0)
    mean_mape_overall = np.sum(list_mae_VAR) / np.sum(gt)

    return forecast



def LSTM(hopsitalization, input_length, output_length, hidden_dim,index_2_fips, fips_2_state):
    scaler = MinMaxScaler()
    hopsitalization = scaler.fit_transform(hopsitalization)

    input_dim = hopsitalization.shape[1] # 49
    output_dim = output_length * input_dim # 4 * 49

    length = hopsitalization.shape[0]
    train_idx = int(length * 0.8)
    val_idx = length - 4
    # Split data into training, validation, and test sets
    # [0-97] as training data, 98 timestamps, [98:118] as validation data, 21 timestamps , [102,122] total 21 timestamps
    train_data = hopsitalization[:train_idx, :]
    val_data = hopsitalization[train_idx:val_idx, :]
    test_data = hopsitalization[(train_idx + 4):, :]

    X_train, y_train = create_dataset(train_data, input_length, output_length)
    X_val, y_val = create_dataset(val_data, input_length, output_length)
    X_test, y_test = create_dataset(test_data, input_length, output_length)

    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    test_dataset = TensorDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=len(val_dataset))
    test_loader = DataLoader(test_dataset, batch_size=len(test_dataset))

    model = LSTMModel(input_dim, hidden_dim, output_dim)

    # Define loss and optimizer
    loss_function = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

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
        training_loss.append(loss.item())
        val_losses.append(val_loss)
    # Testing
    model.eval()
    with torch.no_grad():
        test_loss = 0
        for X_batch, y_batch in test_loader:
            predictions = model(X_batch)
            test_loss += loss_function(predictions, y_batch).item()
        test_loss /= len(test_loader)
        print(f"Test Loss: {test_loss}")
    predictions_reshaped = predictions.reshape(-1, 49).detach().numpy()
    predictions_original_scale = scaler.inverse_transform(predictions_reshaped)
    predictions_original_scale = predictions_original_scale.reshape(-1, 4, 49)
    last_output = predictions_original_scale[-1]

    targeted_label = y_batch[-1]

    loss_of_each_state = abs(targeted_label - last_output)

    LSTM_result = dict()

    for i in range(loss_of_each_state.shape[1]):
        cur_fips = index_2_fips[i]
        cur_state = fips_2_state[cur_fips]
        LSTM_result[cur_state] = torch.sum(loss_of_each_state[:,i]) / 4
    return LSTM_result

def GRU(hopsitalization, input_length, output_length, hidden_dim,index_2_fips, fips_2_state):
    scaler = MinMaxScaler()
    hopsitalization = scaler.fit_transform(hopsitalization)

    input_dim = hopsitalization.shape[1]  # 49
    output_dim = output_length * input_dim  # 4 * 49

    length = hopsitalization.shape[0]
    train_idx = int(length * 0.8)
    val_idx = length - 4
    # Split data into training, validation, and test sets
    # [0-97] as training data, 98 timestamps, [98:118] as validation data, 21 timestamps , [102,122] total 21 timestamps
    train_data = hopsitalization[:train_idx, :]
    val_data = hopsitalization[train_idx:val_idx, :]
    test_data = hopsitalization[(train_idx + 4):, :]

    X_train, y_train = create_dataset(train_data, input_length, output_length)
    X_val, y_val = create_dataset(val_data, input_length, output_length)
    X_test, y_test = create_dataset(test_data, input_length, output_length)

    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    test_dataset = TensorDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=len(val_dataset))
    test_loader = DataLoader(test_dataset, batch_size=len(test_dataset))

    model = GRUModel(input_dim, hidden_dim, output_dim)

    # Define loss and optimizer
    loss_function = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

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
        training_loss.append(loss.item())
        val_losses.append(val_loss)
    # Testing
    model.eval()
    with torch.no_grad():
        test_loss = 0
        for X_batch, y_batch in test_loader:
            predictions = model(X_batch)
            test_loss += loss_function(predictions, y_batch).item()
        test_loss /= len(test_loader)
        print(f"Test Loss: {test_loss}")
    predictions_reshaped = predictions.reshape(-1, 49).detach().numpy()
    predictions_original_scale = scaler.inverse_transform(predictions_reshaped)
    predictions_original_scale = predictions_original_scale.reshape(-1, 4, 49)
    last_output = predictions_original_scale[-1]

    targeted_label = y_batch[-1]

    loss_of_each_state = abs(targeted_label - last_output)

    GRU_result = dict()

    for i in range(loss_of_each_state.shape[1]):
        cur_fips = index_2_fips[i]
        cur_state = fips_2_state[cur_fips]
        GRU_result[cur_state] = torch.sum(loss_of_each_state[:, i]) / 4
    return GRU_result

def DCRNNmodel(hopsitalization, hidden_dim, edge_index):
    scaler = MinMaxScaler()
    hopsitalization = scaler.fit_transform(hopsitalization)

    node_features = 1
    num_nodes = 49

    length = hopsitalization.shape[0]
    train_idx = int(length * 0.8)
    val_idx = length - 4
    # Split data into training, validation, and test sets
    # [0-97] as training data, 98 timestamps, [98:118] as validation data, 21 timestamps , [102,122] total 21 timestamps
    hopsitalization = np.expand_dims(hopsitalization, -1)
    train_data = hopsitalization[:train_idx, :]
    val_data = hopsitalization[train_idx:val_idx, :]
    test_data = hopsitalization[(train_idx + 4):, :]

    X_train, y_train = generate_sequences(train_data, input_length, output_length)
    X_val, y_val = generate_sequences(val_data, input_length, output_length)
    X_test, y_test = generate_sequences(test_data, input_length, output_length)

    # Hyperparameters
    learning_rate = 0.001
    out_channels = 16
    filter_size = 2
    epochs = 10

    model = TrafficPredictor(1, out_channels, filter_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    for epoch in range(epochs):
        model.train()

        # Training Loop
        train_losses = []
        for i in range(len(X_train)):
            optimizer.zero_grad()
            predictions = model(X_train[i], edge_index)
            loss = criterion(predictions, y_train[i])
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        # Validation Loop
        model.eval()
        valid_losses = []
        with torch.no_grad():
            for i in range(len(X_val)):
                predictions = model(X_val[i], edge_index)
                loss = criterion(predictions, y_val[i])
                valid_losses.append(loss.item())

        print(
            f"Epoch {epoch + 1}/{epochs}, Training Loss: {sum(train_losses) / len(train_losses):.4f}, Validation Loss: {sum(valid_losses) / len(valid_losses):.4f}")




def VAR():
    forecast = Vector_Autoregression()




if __name__ == '__main__':
    hopsitalization = pd.read_csv('hosp_weekly_filt_case_data.csv')
    hopsitalization = hopsitalization.reset_index()
    hopsitalization['state'] = hopsitalization['state'].astype(int)

    fips_list = hopsitalization['state'].values
    fips_2_index = hopsitalization.set_index('state')['index'].to_dict()
    index_2_fips = hopsitalization.set_index('index')['state'].to_dict()

    neighbouring_states = pd.read_csv('neighbors-states.csv')
    states_code = pd.read_csv('state_hhs_map.csv')

    neighbouring_states_copy = neighbouring_states[['NeighborStateCode', 'StateCode']]
    neighbouring_states_copy.columns = ['StateCode', 'NeighborStateCode']

    neighbouring_states = neighbouring_states.append(neighbouring_states_copy)
    neighbouring_states.drop_duplicates()

    fips_states = pd.read_csv('state_hhs_map.csv', header=None)
    fips_2_state = fips_states.set_index(0)[2].to_dict()
    state_2_fips = fips_states.set_index(2)[0].to_dict()
    hopsitalization['state'] = hopsitalization['state'].map(fips_2_state)

    index_2_state = hopsitalization.set_index('index')['state'].to_dict()
    state_2_index = hopsitalization.set_index('state')['index'].to_dict()

    neighbouring_states['StateCode'] = neighbouring_states['StateCode'].map(state_2_index)
    neighbouring_states['NeighborStateCode'] = neighbouring_states['NeighborStateCode'].map(state_2_index)

    G = nx.from_pandas_edgelist(neighbouring_states, 'StateCode', 'NeighborStateCode')

    hopsitalization = hopsitalization.iloc[:, 30:]  # remove all 0 datapoints
    hopsitalization = hopsitalization.T.values
    models = ['GRAPHLSTM']

    input_length = 12
    output_length = 4
    hidden_dim = 8

    for model in models:
        if model == 'LSTM':
            lstm_mae_result = LSTM(hopsitalization, input_length, output_length, hidden_dim, index_2_fips, fips_2_state)
        elif model == 'GRU':
            gru_mae_result = GRU(hopsitalization, input_length, output_length, hidden_dim, index_2_fips, fips_2_state)
        elif model == 'DCRNN':
            edge_index = torch.from_numpy(neighbouring_states.values.T)
            DCRNNmodel(hopsitalization, hidden_dim, edge_index)
        else:
            VAR()



