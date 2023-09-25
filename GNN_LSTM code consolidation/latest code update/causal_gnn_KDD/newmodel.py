import json
from copy import deepcopy

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from torch_geometric.data import InMemoryDataset
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch.nn as nn
from sklearn.manifold import TSNE

from covid_recurrent_looping_LSTM import COVIDDatasetSpaced, params
from rnn import LSTM
from weight_sage import WeightedSAGEConv
from torch_geometric.nn import GraphConv

# Define the number of previous and future time steps to use
num_prev_time_steps = 12
num_future_time_steps = 4
# Define the hidden dimension size
hidden_dim = 16
# Define the learning rate and number of epochs to train for
lr = 0.01
num_epochs = 50
dropout_rate = 0.5
l2_penalty = 0.1


# In this script, we only check the status of VA, GA, SC, NC, TN, and we will extract those out


class COVIDDatasetSpaced(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(COVIDDatasetSpaced, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return ['covid_dataset_spaced.dataset']

    def download(self):
        pass

    def process(self):

        # Alternative mobility dataset: Facebook Social Connectedness Index

        data_list = []
        source_nodes = []
        target_nodes = []
        col_names = list(hopsitalization.columns)

        for index, state in enumerate(col_names):
            sub_state = neighbouring_states[neighbouring_states['StateCode'] == state]
            target_states = sub_state['NeighborStateCode'].values

            for t in target_states:
                if t in col_names:
                    source_nodes.append(index)
                    target_nodes.append(col_names.index(t))
                else:
                    continue

        edge_attr = list()

        for i in range(len(source_nodes)):
            edge_attr.append([torch.tensor(1)])
        edge_attr = torch.tensor(edge_attr)
        torch_def = torch.cuda if torch.cuda.is_available() else torch

        params.edge_count = len(source_nodes)

        for i in tqdm(range(len(hopsitalization) - params.lookback_pattern[0])):

            # Node Features
            values_x = []
            for n in params.lookback_pattern:
                m = i + params.lookback_pattern[0] - n
                temp = [np.asarray([hopsitalization.iloc[m, j]], dtype='float64') for j in range(n_states)]
                values_x.append(np.asarray(temp, dtype='float64'))
            values_x = np.asarray(values_x)
            x = torch_def.FloatTensor(values_x)

            # Labels
            values_y = hopsitalization.iloc[(i + params.lookback_pattern[0] + 1):(
                        i + params.lookback_pattern[0] + num_future_time_steps + 1),
                       :].to_numpy().T
            values_y = np.asarray(values_y, dtype='float64')
            y = torch_def.FloatTensor(values_y)

            if y.shape[1] != num_future_time_steps:
                break

            # Edge Index
            nodes = [0, 1, 2, 3, 4]
            edge_index = torch_def.LongTensor([source_nodes.copy(), target_nodes.copy()])
            edge_index = torch_def.LongTensor([nodes.copy(), nodes.copy()])

            data = Data(x=x, edge_index=edge_index, y=y, edge_attr=edge_attr[:5, ])
            data_list.append(data)
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


# Define the GraphLSTM model
class GraphLSTMModel(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, module=WeightedSAGEConv, skip_connection=False,
                 l2_penalty=None
                 , dropout_rate=None):
        super(GraphLSTMModel, self).__init__()
        self.graphsage = WeightedSAGEConv(input_dim, hidden_dim)
        self.lstm1 = LSTM(in_channels=hidden_dim, out_channels=hidden_dim, module=module)
        self.lstm2 = LSTM(in_channels=hidden_dim, out_channels=hidden_dim, module=module)
        self.skip_connection = skip_connection
        self.dropout_rate = dropout_rate
        self.dropout = torch.nn.Dropout(dropout_rate)
        self.act = nn.ReLU()
        self.l2_penalty = l2_penalty
        if skip_connection:
            self.linear = torch.nn.Linear(hidden_dim * 2, output_dim)
        else:
            self.linear = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, data, h=None, c=None):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        x = self.graphsage(x, edge_index, edge_attr)
        embeddings = x.detach().cpu().numpy()
        x = self.act(x)
        x = self.dropout(x)
        h1, c = self.lstm1(x, edge_index, edge_attr, h, c)
        h1 = self.dropout(h1)
        h2, c = self.lstm2(x, edge_index, edge_attr, h1, c)
        h2 = self.dropout(h2)
        if self.skip_connection:
            x = torch.cat((x, h2), dim=2)
        x = self.linear(x[-num_future_time_steps:])

        return x, h, c, embeddings


if __name__ == '__main__':
    # Create an instance of the model
    df2 = pd.read_csv("country_centroids.csv")
    states = pd.read_csv('covid-data/state_hhs_map.csv')
    # VA: 51
    # SC: 45
    # NC: 37
    # GA: 13
    # TN: 47

    selected_states = ['VA', 'SC', 'NC', 'GA', 'TN']
    selected_fips = [51, 45, 37, 13, 47]

    columns = ['name_long', 'Longitude', 'Latitude', 'continent']
    df2 = df2.filter(columns)
    # Choose only a single continent for smaller testing (can choose any continent)
    df2 = df2[df2.continent == 'Europe']

    hopsitalization = pd.read_csv('covid-data/hosp_weekly_filt_case_data.csv')
    hopsitalization = hopsitalization.iloc[:-1, :]
    hopsitalization = hopsitalization.T
    hopsitalization.columns = hopsitalization.iloc[0, :]
    hopsitalization = hopsitalization.iloc[1:, :]
    hopsitalization = hopsitalization.iloc[28:, :]

    # neighboring states
    neighbouring_states = pd.read_csv('covid-data/neighbors-states.csv')
    states_code = pd.read_csv('covid-data/state_hhs_map.csv')

    neighbouring_states_copy = neighbouring_states[['NeighborStateCode', 'StateCode']]
    neighbouring_states_copy.columns = ['StateCode', 'NeighborStateCode']

    neighbouring_states = neighbouring_states.append(neighbouring_states_copy)
    neighbouring_states.drop_duplicates()

    states_code.columns = ['FIPS', 'NUM', 'ABB', 'FULLNAME']
    states_code = states_code.iloc[:, [0, 2]]
    states_code = states_code.iloc[:-3, :]
    n_states = len(selected_states)
    states_map = states_code.set_index('ABB').T.to_dict('list')
    states_fips_map_index = states_code.set_index('FIPS').T.to_dict('list')

    source_nodes = []
    target_nodes = []

    col = hopsitalization.columns
    new_col = list()
    for fips in col:
        new_col.extend(states_fips_map_index[int(fips)])
    hopsitalization.columns = new_col
    hopsitalization = hopsitalization[selected_states]
    dataset = COVIDDatasetSpaced(root='data/covid-data/')

    # train a new model for every new timestamp
    timestamps = np.arange(4, len(dataset), 4)
    candidate_states = selected_states
    states = list(hopsitalization.columns)
    final_looping_prediction = dict()

    best_loss = float('inf')
    num_samples_dropout = 50
    filepath = 'best_model_weights.pt'
    for new_time in tqdm(timestamps):
        train_dataset = dataset[:int(new_time * 0.8)]
        val_dataset = dataset[int(new_time * 0.8):new_time]
        testing_dataset = dataset[new_time]
        model = GraphLSTMModel(input_dim=1, hidden_dim=16, output_dim=1, module=WeightedSAGEConv, skip_connection=True,
                               dropout_rate=dropout_rate, l2_penalty=l2_penalty)
        # Define the loss function and optimizer
        criterion = torch.nn.L1Loss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=model.l2_penalty)

        # Train the model
        samples_50 = list()
        for epoch in range(num_epochs):
            model.train()
            optimizer.zero_grad()
            # Select the previous and future time steps
            h = None
            c = None
            # Forward pass
            loss = 0
            for time, snapshot in enumerate(train_dataset):
                y_pred, h, c, embeddings = model(snapshot, h, c)
                y_pred_update = torch.squeeze(y_pred)
                if snapshot.y.shape[1] == 1:
                    label_transform = torch.squeeze(torch.transpose(snapshot.y, 1, 0))
                else:
                    label_transform = torch.transpose(snapshot.y, 1, 0)
                # Compute loss
                loss = loss + criterion(y_pred_update, label_transform)
                # Take average of loss from all training examples
            loss /= time + 1
            loss.backward()

            # Update parameters
            optimizer.step()

            # Print loss
            # print(f"Epoch {epoch}, Loss: {loss.item()}")

            with torch.no_grad():
                model.eval()

                val_loss = 0
                for val_time, val_snapshot in enumerate(val_dataset):
                    val_y_pred, h, c, embeddings = model(val_snapshot)
                    val_y_pred_update = torch.squeeze(val_y_pred)
                    if val_snapshot.y.shape[1] == 1:
                        val_label_transform = torch.squeeze(torch.transpose(val_snapshot.y, 1, 0))
                    else:
                        val_label_transform = torch.transpose(val_snapshot.y, 1, 0)

                    # Compute loss
                    val_loss = val_loss + criterion(val_y_pred_update, val_label_transform)
                    # Take average of loss from all training examples
                val_loss /= val_time + 1
                # print(f"Epoch {epoch}, Validation Loss: {val_loss.item()}")
            print(
                'Epoch: {:03d}, Train Loss: {:.5f}, Val Loss: {:.5f}'.format(
                    epoch,
                    float(loss.item()),
                    val_loss.item()))

            if val_loss < best_loss:
                best_loss = val_loss
                torch.save(model.state_dict(), filepath)
            # Loop prediction
            if epoch == 49:
                best_model = GraphLSTMModel(input_dim=1, hidden_dim=16, output_dim=1, module=WeightedSAGEConv, skip_connection=True,
                               dropout_rate=dropout_rate, l2_penalty=l2_penalty)
                best_model.load_state_dict(torch.load('best_model_weights.pt'))
                # next_4 = list()
                next_val_4 = list()
                model.train()  # enable dropout in forecasting stage
                val_snapshot = testing_dataset
                x_val_prev = val_snapshot.x
                prediction_dropout = list()
                for num in range(num_samples_dropout):
                    y_val_pred, h, c, embeddings = best_model(val_snapshot)
                    prediction_dropout.append(y_val_pred)
                prediction_stacked = torch.stack(prediction_dropout)
                mean_prediction = torch.mean(prediction_stacked, dim=0)
                uncertainty = torch.std(prediction_stacked, dim=0)
                model.eval()  # turn off dropout
                samples_50 = prediction_dropout

                for can in candidate_states:
                    can_idx = states.index(can)
                    list_of_4_time = list()
                    if new_time == timestamps[0]:
                        final_looping_prediction[can] = dict()
                        final_looping_prediction[can]['recursive_next_4_prediction'] = list()
                        final_looping_prediction[can]['labels'] = list()
                    for j in range(num_future_time_steps):
                        j_th = list()
                        for index, each in enumerate(samples_50):
                            j_th.append(each[j, can_idx, 0].item())
                        list_of_4_time.append(j_th)
                    final_looping_prediction[can]['recursive_next_4_prediction'].append(list_of_4_time)
                    final_looping_prediction[can]['labels'].append(
                        [each_value.item() for each_value in val_snapshot.y[can_idx]])

    with open(
            "IDENTITY_5_states_weekly_recrusive_predictions_vs_labels_4_week_ahead_regularized_12_weeks_mean_variance.json",
            "w") as outfile:
        json.dump(final_looping_prediction, outfile)


