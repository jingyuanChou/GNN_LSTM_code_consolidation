import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, GatedGraphConv
from torch.utils.data import Dataset
import pandas as pd
import json

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from torch_geometric.data import InMemoryDataset
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch.nn as nn

import util
from covid_recurrent_looping_LSTM import COVIDDatasetSpaced, params
from rnn import LSTM
from weight_sage import WeightedSAGEConv
from torch_geometric.nn import GraphConv

def process():
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

    for i in tqdm(range(len(hopsitalization) - 4)):

        # Node Features
        values_x = []
        temp = [np.asarray([hopsitalization.iloc[:i, j]], dtype='float64') for j in range(n_states)]
        values_x.append(np.asarray(temp, dtype='float64'))
        x = torch_def.FloatTensor(values_x)
        x = torch.squeeze(x, dim=0)
        x = x.permute(2,0,1)

        # Labels
        values_y = hopsitalization.iloc[i:(i + 4), :].to_numpy().T
        values_y = np.asarray(values_y, dtype='float64')
        y = torch_def.FloatTensor(values_y)

        if y.shape[1] != 4:
            break

        # Edge Index
        edge_index = torch_def.LongTensor([source_nodes.copy(), target_nodes.copy()])

        data = Data(x=x, edge_index=edge_index, y=y, edge_attr=edge_attr)
        data_list.append(data)
    return data_list


# Define the GraphLSTM model
class GraphLSTMModel(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, module=WeightedSAGEConv, skip_connection=False,
                 l2_penalty=None
                 , dropout_rate=None):
        super(GraphLSTMModel, self).__init__()
        self.graphsage = WeightedSAGEConv(input_dim, hidden_dim)
        self.lstm = LSTM(in_channels=hidden_dim, out_channels=hidden_dim, module=module)
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
        h, c = self.lstm(x, edge_index, edge_attr, h, c)
        if self.skip_connection:
            x = torch.cat((x, h), dim=2)
        x = self.dropout(x)
        x = self.linear(x[-1:])

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

    dataset = process()

    # train a new model for every new timestamp
    timestamps = np.arange(4, len(dataset), 4)
    candidate_states = selected_states
    states = list(hopsitalization.columns)
    final_looping_prediction = dict()

    # plots

    reader = open('5_states_weekly_recrusive_predictions_vs_labels_4_week_ahead_autoregressive_training.json')
    one_week = json.load(reader)

    for can in candidate_states:
        all_values_single_state = hopsitalization[can].values
        plt.plot(range(len(all_values_single_state)), all_values_single_state, label='{} labels'.format(can),
                 color='black')
        for index, time in enumerate(timestamps):
            if index == len(timestamps) - 1:
                break
            starting_point =  time
            # x_axis = range(starting_point - 3 ,starting_point+1)
            x_axis = range(starting_point, starting_point + 4)

            if index == 0:
                means = [np.mean(d) for d in one_week[can]['recursive_next_4_prediction'][index]]
                std = [np.std(d) for d in one_week[can]['recursive_next_4_prediction'][index]]
                plt.plot(x_axis, means, label='{} predictions'.format(can), color='red', linestyle='--', marker='o',
                         markersize=2.0)
                plt.fill_between(x_axis, np.subtract(means, std), np.add(means, std), alpha=0.6)
                total_loss = 0
                for j in range(4):
                    total_loss = total_loss + np.abs(means[j] - one_week[can]['labels'][index][j])
                print(x_axis, total_loss / 4)
            else:
                means = [np.mean(d) for d in one_week[can]['recursive_next_4_prediction'][index]]
                std = [np.std(d) for d in one_week[can]['recursive_next_4_prediction'][index]]
                plt.plot(x_axis, means,
                         color='red', linestyle='--', marker='o', markersize=2.0)
                plt.fill_between(x_axis, np.subtract(means, std), np.add(means, std), alpha=0.2)
                total_loss = 0
                for j in range(4):
                    total_loss = total_loss + np.abs(means[j] - one_week[can]['labels'][index][j])
                print(x_axis, total_loss / 4)
        plt.title('{} prediction'.format(can))
        plt.legend()
        plt.show()


    '''
        reader = open('5_states_weekly_recrusive_predictions_vs_labels_4_week_ahead_regularized_12_weeks_mean_variance.json')
        one_week = json.load(reader)

        for can in candidate_states:
            all_values_single_state = hopsitalization[can].values
            plt.plot(range(len(all_values_single_state)), all_values_single_state, label='{} labels'.format(can), color='black')
            for index, time in enumerate(timestamps):
                starting_point = num_prev_time_steps + time
                # x_axis = range(starting_point - 3 ,starting_point+1)
                x_axis = range(starting_point,starting_point+4)

                if index == 0:
                    means = [np.mean(d) for d in one_week[can]['recursive_next_4_prediction'][index]]
                    std = [np.std(d) for d in one_week[can]['recursive_next_4_prediction'][index]]
                    plt.plot(x_axis, means, label='{} predictions'.format(can), color='red',  linestyle='--', marker='o', markersize = 2.0)
                    plt.fill_between(x_axis, np.subtract(means, std), np.add(means, std), alpha=0.6)
                else:
                    means = [np.mean(d) for d in one_week[can]['recursive_next_4_prediction'][index]]
                    std = [np.std(d) for d in one_week[can]['recursive_next_4_prediction'][index]]
                    plt.plot(x_axis, means,
                              color='red', linestyle='--', marker='o', markersize = 2.0)
                    plt.fill_between(x_axis, np.subtract(means, std), np.add(means, std), alpha=0.2)
            plt.title('{} prediction'.format(can))
            plt.legend()
            plt.show()
        '''

    model = GraphLSTMModel(input_dim=1, hidden_dim=16, output_dim=1, module=WeightedSAGEConv, skip_connection=True,
                           dropout_rate=0.2, l2_penalty=0.01)
    # Define the loss function and optimizer

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=model.l2_penalty)
    num_epochs = 500
    # Train the model
    orginal_data = dataset.copy()
    for time in tqdm(timestamps):
        if time == timestamps[-1]:
            break
        label = orginal_data[time].y
        pred_epochs = list()
        for epoch in range(num_epochs):
            pred_list = list()
            loss_epochs = list()
            per_loss = 0
            for i in range(time,time+4):
                if time + 4 >= len(dataset):
                    break
                if i == 0:
                    continue
            # Get the input and target data for this timestamp
                data = dataset[i]
                x, y_true = data.x, data.y
                # Zero the gradients and perform a forward pass
                optimizer.zero_grad()
                y_pred,h,c = model(data)

                # Compute the loss and perform backpropagation
                loss = util.MAE_loss(y_pred, y_true)
                loss.backward()
                optimizer.step()
                pred_list.append(torch.squeeze(y_pred,dim=0))


                # Set the input for the next timestamp to be the output of the current timestamp
                if i < len(dataset) - 1:
                    dataset[i + 1].x[-1, :, :] = y_pred.detach()
            per_loss = per_loss + loss.item()
            if len(pred_list) != 0:
                pred_list = torch.stack(pred_list, dim=0)
            else:
                break
            # print(f"Epoch {epoch + 1}, Loss: {per_loss}")

            #per_loss = per_loss / (len(dataset) - 1) # average loss
            pred_epochs.append(pred_list)
            # Print the loss for this epoch
        for can in candidate_states:
            can_idx = states.index(can)
            list_of_4_time = list()
            if time == timestamps[0]:
                final_looping_prediction[can] = dict()
                final_looping_prediction[can]['recursive_next_4_prediction'] = list()
                final_looping_prediction[can]['labels'] = list()
            for j in range(4):
                j_th = list()
                list_of_4_time.append(j_th)
                for index, each in enumerate(pred_epochs):
                    if index < num_epochs:
                        j_th.append(each[j, can_idx, 0].item())
            final_looping_prediction[can]['recursive_next_4_prediction'].append(list_of_4_time)
            final_looping_prediction[can]['labels'].append(
                [label[can_idx, idx].item() for idx in range(0, 4)])


    with open("5_states_weekly_recrusive_predictions_vs_labels_4_week_ahead_autoregressive_training.json", "w") as outfile:
        json.dump(final_looping_prediction, outfile)


