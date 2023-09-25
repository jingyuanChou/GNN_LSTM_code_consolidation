import json

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from torch_geometric.data import InMemoryDataset
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch.nn as nn

from covid_recurrent_looping_LSTM import COVIDDatasetSpaced, params
from rnn import LSTM
from weight_sage import WeightedSAGEConv
from torch_geometric.nn import GraphConv

# Define the number of previous and future time steps to use
num_prev_time_steps = 12
num_future_time_steps = 4
# Define the hidden dimension size
hidden_dim = 4
# Define the learning rate and number of epochs to train for
lr = 0.01
num_epochs = 1
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
        col_names = list(states_cases.columns)

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

        for i in tqdm(range(len(states_cases) - params.lookback_pattern[0])):

            # Node Features
            values_x = []
            for n in params.lookback_pattern:
                m = i + params.lookback_pattern[0] - n
                temp = [np.asarray([states_cases.iloc[m, j]], dtype='float64') for j in range(n_states)]
                values_x.append(np.asarray(temp, dtype='float64'))
            values_x = np.asarray(values_x)
            x = torch_def.FloatTensor(values_x)

            # Labels
            values_y = states_cases.iloc[(i + params.lookback_pattern[0] + 1):(i + params.lookback_pattern[0] + num_future_time_steps + 1),
                       :].to_numpy().T
            values_y = np.asarray(values_y, dtype='float64')
            y = torch_def.FloatTensor(values_y)

            if y.shape[1] != num_future_time_steps:
                break

            # Edge Index
            edge_index = torch_def.LongTensor([source_nodes.copy(), target_nodes.copy()])

            data = Data(x=x, edge_index=edge_index, y=y, edge_attr=edge_attr)
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
        x = self.act(x)
        x = self.dropout(x)
        h1, c = self.lstm1(x, edge_index, edge_attr, h, c)
        h1 = self.dropout(h1)
        h2, c = self.lstm2(x, edge_index, edge_attr, h1, c)
        h2 = self.dropout(h2)
        if self.skip_connection:
            x = torch.cat((x,h2), dim=2)
        x = self.linear(x[-num_future_time_steps:])

        return x, h, c


if __name__ == '__main__':
    # Create an instance of the model
    df2 = pd.read_csv("country_centroids.csv")
    states = pd.read_csv('/Users/bijiehao/Downloads/causal_gnn_KDD/covid-data/state_hhs_map.csv')
    states_cases = pd.read_csv('/Users/bijiehao/Downloads/causal_gnn_KDD/covid-data/weekly_filt_case_state_level_2020_2022.csv')
    # VA: 51
    # SC: 45
    # NC: 37
    # GA: 13
    # TN: 47

    selected_states = ['VA', 'SC', 'NC', 'GA', 'TN']
    selected_fips = [51, 45, 37, 13, 47]

    # neighboring states
    neighbouring_states = pd.read_csv('/Users/bijiehao/Downloads/causal_gnn_KDD/covid-data/neighbors-states.csv')
    states_code = pd.read_csv('/Users/bijiehao/Downloads/causal_gnn_KDD/covid-data/state_hhs_map.csv')

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

    states_cases = states_cases.T
    states_cases.columns = ['GA','NC','SC','TN','VA']
    states_cases = states_cases.iloc[1:,:]

    dataset = COVIDDatasetSpaced(root='covid-data/')

    # train a new model for every new timestamp
    timestamps = np.arange(1, len(dataset))
    candidate_states = selected_states
    states = list(states_cases.columns)
    final_looping_prediction = dict()

    # plots

    '''
    reader = open('5_states_weekly_cases_vs_labels_4_week_ahead_regularized_12_weeks_mean_variance.json')
    one_week = json.load(reader)

    for can in candidate_states:
        all_values_single_state = states_cases[can].values
        plt.plot(range(len(all_values_single_state)), all_values_single_state, label='{} labels'.format(can),
                 color='black')
        for index, time in enumerate(timestamps):
            starting_point = num_prev_time_steps + time - 4
            # x_axis = range(starting_point - 3 ,starting_point+1)
            x_axis = np.arange(starting_point,starting_point+4)

            if index == 0:
                means = [np.mean(d) for d in one_week[can]['recursive_next_4_prediction'][index]]
                std = [np.std(d) for d in one_week[can]['recursive_next_4_prediction'][index]]
                plt.plot(x_axis, means, label='{} predictions'.format(can), color='red',  linestyle='--', marker='o', markersize = 2.0)
                plt.fill_between(x_axis, np.subtract(means, std), np.add(means, std), alpha=0.6)

            else:
                means = [np.mean(d) for d in one_week[can]['recursive_next_4_prediction'][index]]
                std = [np.std(d) for d in one_week[can]['recursive_next_4_prediction'][index]]
                plt.plot(x_axis, means,
                         color='red', linestyle='--', marker='o', markersize=2.0)
                plt.fill_between(x_axis, np.subtract(means, std), np.add(means, std), alpha=0.2)

                total_loss = 0
                for j in range(4):
                    total_loss = total_loss + np.abs(means[j] - one_week[can]['labels'][index - 1][j])
                print(x_axis, total_loss / 4, means, std, np.array(std) / np.array(means))

        plt.title('{} prediction'.format(can))
        plt.legend()
        plt.show()
    '''
    num_samples_dropout = 50
    for new_time in tqdm(timestamps):
        if new_time == timestamps[-1]:
            print('dw')
        train_dataset = dataset[:new_time]
        val_dataset = dataset[new_time:]
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

                y_pred, h, c = model(snapshot, h, c)

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
                    val_y_pred, h, c = model(val_snapshot)
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

            # Loop prediction
            if epoch == 49:
                # next_4 = list()
                next_val_4 = list()
                with torch.no_grad():
                    model.train()
                    val_snapshot = val_dataset[0]
                    x_val_prev = val_snapshot.x
                    prediction_dropout = list()
                    for num in range(num_samples_dropout):
                        y_val_pred, h, c = model(val_snapshot)
                        prediction_dropout.append(y_val_pred)
                    prediction_stacked = torch.stack(prediction_dropout)
                    mean_prediction = torch.mean(prediction_stacked, dim=0)
                    uncertainty = torch.std(prediction_stacked, dim=0)
                    model.eval()  # turn off dropout
                    samples_50 = prediction_dropout
                    '''
                    y_val_pred_loop = y_val_pred  # next 1 step
                    y_val_pred_loop = torch.unsqueeze(y_val_pred_loop[0], dim=0)
                    next_val_4.append(y_val_pred_loop)
                    for i in range(num_future_time_steps - 1):
                        # next 2, 3, 4 steps
                        x_val_prev = torch.cat((x_val_prev[1:], y_val_pred_loop))
                        y_val_pred_loop, _, _ = model(
                            Data(x=x_val_prev, edge_index=val_snapshot.edge_index, edge_attr=val_snapshot.edge_attr))
                        y_val_pred_loop = torch.unsqueeze(y_val_pred_loop[0], dim=0)
                        next_val_4.append(y_val_pred_loop)
                samples_50.append(next_val_4)
                '''

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

    with open("5_states_weekly_cases_vs_labels_4_week_ahead_regularized_12_weeks_mean_variance.json",
                  "w") as outfile:
            json.dump(final_looping_prediction, outfile)
