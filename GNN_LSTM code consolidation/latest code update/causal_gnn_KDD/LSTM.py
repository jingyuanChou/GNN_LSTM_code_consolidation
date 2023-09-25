import torch
from torch import nn
from torch.nn import functional as F

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from scipy.signal import savgol_filter
from datetime import datetime, timedelta
from sklearn.cluster import KMeans

import argparse
import re
import torch
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt


class lstm_mcdropout(nn.Module):
    def __init__(self, input_shape, hidden_rnn, hidden_dense, output_dim, activation):
        super().__init__()
        self.lstm = nn.LSTM(input_shape, hidden_rnn, batch_first=True)
        self.dense = nn.Linear(hidden_rnn, hidden_dense)
        self.dropout = nn.Dropout(0.2)
        self.output_layer = nn.Linear(hidden_dense, output_dim)
        self.activation = activation
        if activation == 'relu':
            self.act = nn.ReLU()
        # Implement other activation functions if required

    def forward(self, x):
        num_samples = 5
        x, _ = self.lstm(x)
        pred_list = list()
        for i in range(num_samples):
            x_pred = self.act(self.dense(x[i, -1, :]))  # Take the output from the last time step only
            x_pred = self.dropout(x_pred)
            x_pred = self.output_layer(x_pred)
            pred_list.append(x_pred)
        return pred_list



if __name__ == '__main__':
    df2 = pd.read_csv("country_centroids.csv")
    states = pd.read_csv('covid-data/state_hhs_map.csv')
    # VA: 51
    # SC: 45
    # NC: 37
    # GA: 13
    # TN: 47
    initial_sequence_length = 12
    num_samples_dropout = 50
    forecast_horizon = 4
    num_epochs = 200
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

    hopsitalization = hopsitalization[
        selected_states]  # Assume `hospitalizations` is a numpy array of shape (t, 5), where t is the total number of timestamps
    hopsitalization = hopsitalization.astype(float)

    # Initialize the model
    model = lstm_mcdropout(input_shape=5, hidden_rnn=16, hidden_dense = 50, output_dim=1, activation='relu')
    # Define loss function
    loss_function = torch.nn.L1Loss()
    # Define optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.08, weight_decay=0.001)
    # Initial input sequence and target sequence
    input_sequences = list()
    target_sequences = list()
    final_looping_prediction = dict()
    states = list(hopsitalization.columns)


    for start_index in range(0, hopsitalization.shape[0] - initial_sequence_length - forecast_horizon + 1):
        # Add a new input sequence and target sequence for each new timestamp
        input_sequences.append(hopsitalization[start_index:start_index + initial_sequence_length])
        target_sequences.append(
            hopsitalization[start_index + initial_sequence_length:start_index + initial_sequence_length + forecast_horizon])

        input_sequences_array= np.array(input_sequences)
        target_sequences_array = np.array(target_sequences)

        # Convert the input sequences and target sequences to PyTorch tensors
        input_data = torch.tensor(input_sequences_array, dtype=torch.float32)
        target_data = torch.tensor(target_sequences_array, dtype=torch.float32)
        loss_per_index = list()
        # Train the model for a certain number of epochs on the entire training set
        if start_index != 0 and start_index % 4 == 0:
            forecasting_sequence  = hopsitalization[start_index:start_index + initial_sequence_length]
            forecasting_target = hopsitalization[start_index + initial_sequence_length:start_index + initial_sequence_length + forecast_horizon]
            for epoch in range(num_epochs):
                optimizer.zero_grad()
                output = model(input_data[:,:,0])
                loss = loss_function(output, target_data[:,:,0])
                loss.backward()
                optimizer.step()
                loss_per_index.append(loss.item())
                print(np.mean(loss_per_index))

            next_pred_4 = list()
            model.train()  # enable dropout in forecasting stage
            prediction_dropout = list()
            forecasting_sequence = np.array(forecasting_sequence)
            forecasting_sequence = torch.tensor(forecasting_sequence, dtype=torch.float32)
            forecasting_sequence = forecasting_sequence.unsqueeze(0).float()
            for num in range(num_samples_dropout):
                y_pred, _ = model(forecasting_sequence)
                prediction_dropout.append(y_pred)
            prediction_stacked = torch.stack(prediction_dropout)
            mean_prediction = torch.mean(prediction_stacked, dim=0)
            uncertainty = torch.std(prediction_stacked, dim=0)
            model.eval()  # turn off dropout
            samples_50 = prediction_dropout

            for can in selected_states:
                can_idx = states.index(can)
                list_of_4_time = list()
                if start_index == 4:
                    final_looping_prediction[can] = dict()
                    final_looping_prediction[can]['recursive_next_4_prediction'] = list()
                    final_looping_prediction[can]['labels'] = list()
                for j in range(forecast_horizon):
                    j_th = list()
                    for index, each in enumerate(samples_50):
                        j_th.append(each[0, j, can_idx].item())
                    list_of_4_time.append(j_th)
                final_looping_prediction[can]['recursive_next_4_prediction'].append(list_of_4_time)
                final_looping_prediction[can]['labels'].append(forecasting_target.iloc[:,can_idx].values)

    '''
    with open("LSTM_5_states_weekly_recrusive_predictions_vs_labels_4_week_ahead_regularized_12_weeks_mean_variance.json",
              "w") as outfile:
        json.dump(final_looping_prediction, outfile)
    '''
    timestamps = np.arange(4, len(hopsitalization)-initial_sequence_length, 4)

    candidate_states = selected_states
    MAEP = dict()
    abbre = {'NC':'North Carolina', 'SC':'South Carolina','VA':'Virginia','GA':'Georgia','TN':'Tennessee'}
    for can in candidate_states:
        MAEP[can] = dict()
        all_values_single_state = hopsitalization[can].values
        for index, time in enumerate(timestamps):
            if index >= 26:
                continue
            starting_point = initial_sequence_length + time - 4
            # x_axis = range(starting_point - 3 ,starting_point+1)
            x_axis = np.arange(starting_point + 1,starting_point + 5)
            if index == 0:
                plt.plot(range(1,x_axis[-1] + 1), all_values_single_state[0: x_axis[-1]], label='{} actual'.format(can),
                     color='black')
            else:
                plt.plot(range(1, x_axis[-1] + 1), all_values_single_state[0: x_axis[-1]], color='black')
            if index == 0:
                means = [np.mean(d) for d in final_looping_prediction[can]['recursive_next_4_prediction'][index]]
                std = [np.std(d) for d in final_looping_prediction[can]['recursive_next_4_prediction'][index]]
                plt.plot(x_axis, means, label='{} predictions'.format(can), color='red',  linestyle='--', marker='o', markersize = 2.0)
                plt.fill_between(x_axis, np.subtract(means, std), np.add(means, std), alpha=0.6)
                # calculate MAPE for the forecast at different times
                actual_value = all_values_single_state[0: x_axis[-1]][-4:]
                temp_time_end = starting_point + 4
                temp_time_str = str(starting_point) +' - '+str(temp_time_end)
                difference = abs(means-actual_value)
                MAEP[can][temp_time_str] = str(np.sum(difference)/np.sum(actual_value))

            else:
                means = [np.mean(d) for d in final_looping_prediction[can]['recursive_next_4_prediction'][index]]
                std = [np.std(d) for d in final_looping_prediction[can]['recursive_next_4_prediction'][index]]
                plt.plot(x_axis, means,
                          color='red', linestyle='--', marker='o', markersize = 2.0)
                plt.fill_between(x_axis, np.subtract(means, std), np.add(means, std), alpha=0.2)
                total_loss = 0
                # calculate MAPE for the forecast at different times
                actual_value = all_values_single_state[0: x_axis[-1]][-4:]
                temp_time_end = starting_point + 4
                temp_time_str = str(starting_point) + ' - ' + str(temp_time_end)
                difference = abs(means - actual_value)
                MAEP[can][temp_time_str] = str(np.sum(difference) / np.sum(actual_value))

                for j in range(4):
                    total_loss = total_loss + np.abs(means[j] - final_looping_prediction[can]['labels'][index - 1][j])
                    print(x_axis, total_loss / 4, means, std, np.array(std)/np.array(means))


        plt.title('{} prediction'.format(abbre[can]))
        plt.xlabel('Time (Weeks)')
        plt.ylabel('# Hospitalizations')
        plt.legend()
        plt.show()
    print('OK')




