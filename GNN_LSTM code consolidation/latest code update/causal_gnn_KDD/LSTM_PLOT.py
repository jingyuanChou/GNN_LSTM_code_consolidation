import json
from copy import deepcopy

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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

    hopsitalization = pd.read_csv('covid-data/weekly_filt_case_state_level_2020_2022.csv')
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

    # train a new model for every new timestamp
    timestamps = range(4,71)
    candidate_states = selected_states
    states = list(hopsitalization.columns)
    final_looping_prediction = dict()


    times = range(4,71)
    # plots
    states = ['VA','SC','NC','GA','TN']
    final_looping_prediction = dict()
    for state in states:
        data = pd.read_csv('{}_case_dropout02_LSTM_only.csv'.format(state), header=None)
        new_col = ['fips','date','horizon','mc']
        new_col.extend(times)
        data.columns = new_col
        can_idx = states.index(state)
        final_looping_prediction[state] = dict()
        final_looping_prediction[state]['recursive_next_4_prediction'] = dict()
        for new_time in times:
            final_looping_prediction[state]['recursive_next_4_prediction'][new_time] = list()
            j_th_1 = list()
            j_th_2 = list()
            j_th_3 = list()
            j_th_4 = list()
            for index, each in data.iterrows():
                if each['horizon'] == 0:
                    j_th_1.append(each[new_time])
                elif each['horizon'] == 1:
                    j_th_2.append(each[new_time])
                elif each['horizon'] == 2:
                    j_th_3.append(each[new_time])
                else:
                    j_th_4.append(each[new_time])

            final_looping_prediction[state]['recursive_next_4_prediction'][new_time].append(j_th_1)
            final_looping_prediction[state]['recursive_next_4_prediction'][new_time].append(j_th_2)
            final_looping_prediction[state]['recursive_next_4_prediction'][new_time].append(j_th_3)
            final_looping_prediction[state]['recursive_next_4_prediction'][new_time].append(j_th_4)

    with open("Cases_5_states_weekly_recrusive_predictions_vs_labels_4_week_ahead_regularized_12_weeks_mean_variance.json",
              "w") as outfile:
        json.dump(final_looping_prediction, outfile)

    abbre = {'NC': 'North Carolina', 'SC': 'South Carolina', 'VA': 'Virginia', 'GA': 'Georgia', 'TN': 'Tennessee'}
    MAEP = dict()
    for state in states:
        MAEP[state] = dict()
        all_values_single_state = hopsitalization[state].values
        for index, time in enumerate(timestamps):
            # x_axis = range(starting_point - 3 ,starting_point+1)
            x_axis = np.arange(time + 1, time + 5)
            if index == 0:
                plt.plot(range(1, x_axis[-1] + 1), all_values_single_state[0: x_axis[-1]],
                         label='{} actual'.format(state),
                         color='black')
            else:
                plt.plot(range(1, x_axis[-1] + 1), all_values_single_state[0: x_axis[-1]], color='black')
            if index == 0:
                means = [np.mean(d) for d in final_looping_prediction[state]['recursive_next_4_prediction'][time]]
                std = [np.std(d) for d in final_looping_prediction[state]['recursive_next_4_prediction'][time]]
                plt.plot(x_axis, means, label='{} predictions'.format(state), color='red', linestyle='--', marker='o',
                         markersize=2.0)
                plt.fill_between(x_axis, np.subtract(means, std), np.add(means, std), alpha=0.6)
                # calculate MAPE for the forecast at different times
                actual_value = all_values_single_state[0: x_axis[-1]][-4:]
                temp_time_end = time + 4
                temp_time_str = str(time) + ' - ' + str(temp_time_end)
                difference = abs(means - actual_value)
                MAEP[state][temp_time_str] = str(np.sum(difference) / np.sum(actual_value))

            else:
                means = [np.mean(d) for d in final_looping_prediction[state]['recursive_next_4_prediction'][time]]
                std = [np.std(d) for d in final_looping_prediction[state]['recursive_next_4_prediction'][time]]
                plt.plot(x_axis, means,
                         color='red', linestyle='--', marker='o', markersize=2.0)
                plt.fill_between(x_axis, np.subtract(means, std), np.add(means, std), alpha=0.2)
                total_loss = 0
                # calculate MAPE for the forecast at different times
                actual_value = all_values_single_state[0: x_axis[-1]][-4:]
                temp_time_end = time + 4
                temp_time_str = str(time) + ' - ' + str(temp_time_end)
                difference = abs(means - actual_value)
                MAEP[state][temp_time_str] = str(np.sum(difference) / np.sum(actual_value))

        plt.title('{} prediction'.format(abbre[state]))
        plt.xlabel('Time (Weeks)')
        plt.ylabel('# Hospitalizations')
        plt.legend()
        plt.show()



