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
            values_y = hopsitalization.iloc[(i + params.lookback_pattern[0] + 1):(i + params.lookback_pattern[0] + 5),
                       :].to_numpy().T
            values_y = np.asarray(values_y, dtype='float64')
            y = torch_def.FloatTensor(values_y)

            if y.shape[1] != 4:
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
        self.l2_penalty = l2_penalty
        if skip_connection:
            self.linear = torch.nn.Linear(hidden_dim * 2, output_dim)
        else:
            self.linear = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, data, h=None, c=None):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        x = self.graphsage(x, edge_index, edge_attr)
        h1, c = self.lstm1(x, edge_index, edge_attr, h, c)
        h1 = self.dropout(h1)
        h2, c = self.lstm2(x, edge_index, edge_attr, h1, c)
        if self.skip_connection:
            x = torch.cat((h1, h2), dim=2)
        x = self.linear(x[-num_future_time_steps:])

        return x, h, c


if __name__ == '__main__':
    # Create an instance of the model
    df2 = pd.read_csv("country_centroids.csv")
    states = pd.read_csv('/Users/bijiehao/Downloads/causal_gnn_KDD/covid-data/state_hhs_map.csv')
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

    hopsitalization = pd.read_csv('/Users/bijiehao/Downloads/causal_gnn_KDD/covid-data/hosp_weekly_filt_case_data.csv')
    hopsitalization = hopsitalization.iloc[:-1, :]
    hopsitalization = hopsitalization.T
    hopsitalization.columns = hopsitalization.iloc[0, :]
    hopsitalization = hopsitalization.iloc[1:, :]
    hopsitalization = hopsitalization.iloc[28:, :]

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
    week = 4
    # plots
    reader = open('5_states_weekly_recrusive_predictions_vs_labels_1_week_ahead_regularized_12_weeks_mean_variance.json')
    one_week = json.load(reader)

    reader = open('5_states_weekly_recrusive_predictions_vs_labels_4_week_ahead_regularized_12_weeks_mean_variance.json')
    one_week = json.load(reader)

    for can in candidate_states:
        all_values_single_state = hopsitalization[can].values
        for index, time in enumerate(timestamps):
            if index == 0:
                continue
            starting_point = num_prev_time_steps + time - 4
            # x_axis = range(starting_point - 3 ,starting_point+1)
            if week == 4:
                x_axis = np.arange(starting_point + 1,starting_point+5)
                plt.plot(range(1, x_axis[-1] + 1), all_values_single_state[0: x_axis[-1]],
                         label='{} labels'.format(can),
                         color='black', marker='o', markersize = 3.0)

            means = [np.mean(d) for d in one_week[can]['recursive_next_4_prediction'][index]]
            std = [np.std(d) for d in one_week[can]['recursive_next_4_prediction'][index]]
            plt.plot(x_axis, means, label='{} predictions'.format(can), color='red', linestyle='--', marker='o',
                     markersize=2.0)
            plt.fill_between(x_axis, np.subtract(means, std), np.add(means, std), alpha=0.2)

            plt.title('{} prediction'.format(can))
            plt.legend()
            plt.show()

            total_loss = 0
            if week == 4:
                for j in range(4):
                    total_loss = total_loss + np.abs(means[j] - one_week[can]['labels'][index - 1][j])
            else:
                total_loss = total_loss + np.abs(np.array(means) - np.array(one_week[can]['labels'][index - 1]))
            print(x_axis, total_loss / 4, means, std, np.array(std)/np.array(means), np.subtract(means,all_values_single_state[x_axis]))


