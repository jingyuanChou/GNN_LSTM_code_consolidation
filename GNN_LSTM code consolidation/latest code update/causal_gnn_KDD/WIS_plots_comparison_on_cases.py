import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from copy import deepcopy

# Define the path to the repository where the CSV files are stored
path_to_repo = "result_evaluation/"

# Use os to get a list of all files in the repository
files = os.listdir(path_to_repo)

# Sort the list of files alphabetically
files.sort()



def get_WIS(df):
    '''Computes the WIS score given a forecast dataframe in the standard ForecastHub quantile submission format. This code uses the eq. (4) of Evaluating epidemic forecasts in an interval format'''
    df=df[df['type']!='point']
    df.loc[:,'QS']=(2*((df.truth<=df.value)-df['quantile'])*(df['value']-df.truth))
    wdf=df.groupby(['location','forecast_date','target_end_date','target'],as_index=None).mean().rename(columns={'QS':'WIS'}).drop(['quantile','value'],axis=1)
    return wdf

if __name__ == '__main__':
    '''
    label_WIS = list()
    self_WIS = list()
    LSTM_WIS = list()
    # Loop through the sorted list of files
    for i in range(53):
        if i == 0:
            continue
        # Check for pairs of files with the same date format
        if files[i][0:10] == files[i + 52][-14:-4] == files[i+104][-14:-4]:
            # Read the CSV files using pandas
            file1 = pd.read_csv(path_to_repo + files[i]) # label data
            file2 = pd.read_csv(path_to_repo + files[i + 52]) # self data
            file3 = pd.read_csv(path_to_repo + files[i + 104])

            locations = ['51', '45', '37', '13', '47']
            weeks = ['1 wk ahead inc case', '2 wk ahead inc case', '3 wk ahead inc case', '4 wk ahead inc case']
            quantiles = [0.025, 0.1, 0.25, 0.5, 0.75, 0.9, 0.975]

            file2 = file2.iloc[:, 1:]
            file3 = file3.iloc[:, 1:]
            file1 = file1[file1['location'].isin(locations)]
            file1 = file1[file1['target'].isin(weeks)]
            file1.reset_index(inplace=True)
            file1 = file1.iloc[:, 1:]
            file1['truth'] = 0

            for i in range(len(file1)):
                td = file1.iloc[i]['target_end_date']
                loc = file1.iloc[i]['location']
                sub1 = file2[file2['target_end_date'] == td]
                sub3 = file3[file3['target_end_date'] == td]
                sub2 = sub1[sub1['location'] == int(loc)]
                truth_value = sub2['truth'].values[0]
                file1.iloc[i, 7] = truth_value

            file3 = file3.sort_values(['target_end_date', 'location', 'quantile'], ascending=[True, False, False])
            file2 = file2.sort_values(['target_end_date', 'location', 'quantile'], ascending=[True, False, False])
            file1 = file1.sort_values(['target_end_date', 'location', 'quantile'], ascending=[True, False, False])

            WIS_mine = get_WIS(file2)
            WIS_LSTM_ONLY = get_WIS(file3)
            WIS_ensemble = get_WIS(file1)

            label_WIS.append(WIS_ensemble)
            self_WIS.append(WIS_mine)
            LSTM_WIS.append(WIS_LSTM_ONLY)

            print('processing multiple files...')
    
    with open('labels.pkl', 'wb') as f:
        pd.to_pickle(label_WIS, f)
    f.close()
    with open('self.pkl', 'wb') as g:
        pd.to_pickle(self_WIS, g)
    g.close()
    with open('LSTM_WIS.pkl', 'wb') as m:
        pd.to_pickle(LSTM_WIS, m)
    m.close()
    '''

    with open('labels.pkl', 'rb') as f:
        labels = pd.read_pickle(f)

    with open('self.pkl', 'rb') as g:
        self = pd.read_pickle(g)# Print the first dataframe in the list

    with open('LSTM_WIS.pkl', 'rb') as m:
        LSTM_WIS = pd.read_pickle(m)

    L = list()
    S = list()
    LSTM_ONLY = list()
    import datetime

    start_time = datetime.datetime.strptime('2020-10-05', '%Y-%m-%d').date()
    x = list()
    states = [13,37,45,47,51]
    state_WIS = dict()
    state_WIS_forecast = dict()
    state_WIS_LSTM = dict()
    for index in range(52):
        x.append(start_time)
        start_time = start_time + datetime.timedelta(days= 7)
        for state in states:
            sub_labels = labels[index][labels[index]['location'] == str(state)]
            state_wis_label = np.sum(sub_labels['WIS'].values)

            sub_self_wis = self[index][self[index]['location'] == state]
            state_self_wis = np.sum(sub_self_wis['WIS'].values)

            sub_LSTM_wis = LSTM_WIS[index][LSTM_WIS[index]['location'] == state]
            state_LSTM_wis = np.sum(sub_LSTM_wis['WIS'].values)

            if state not in state_WIS.keys():
                state_WIS[state] = list()
                state_WIS[state].append(state_wis_label)
                state_WIS_forecast[state] = list()
                state_WIS_forecast[state].append(state_self_wis)
                state_WIS_LSTM[state] = list()
                state_WIS_LSTM[state].append(state_LSTM_wis)
            else:
                state_WIS[state].append(state_wis_label)
                state_WIS_forecast[state].append(state_self_wis)
                state_WIS_LSTM[state].append(state_LSTM_wis)

    state_WIS_forecast_LSTM = dict()

    for state in state_WIS_forecast.keys():
        val_list = state_WIS_forecast[state]
        LSTM_VAL_LS = state_WIS_LSTM[state]
    '''
    with open('LSTM_forecast.pkl', 'wb') as f:
        pd.to_pickle(state_WIS_forecast_LSTM, f)
    f.close()
    '''
    state_names = {51:'Virginia',45:'South Carolina', 37:'North Carolina',13:'Georgia',47:'Tennessee'}
    for state in states:
        plt.plot(x, state_WIS[state], label='CovidHub Ensemble', color='black',marker='o', markersize = 3.0, )
        plt.plot(x, state_WIS_forecast[state], label='Our Method', color='red',marker='D', markersize = 2.0)
        plt.plot(x, state_WIS_LSTM[state], label='LSTM Only', color='cyan',marker='>', markersize = 2.0)
        plt.title('WIS Comparison for {}'.format(state_names[state]))
        plt.ylabel('Cumulative WIS')
        plt.xlabel('Time')
        plt.legend()
        plt.show()
        ###### 1st week

    '''
    L = list()
    S = list()
    import datetime
    
    # weeks = ['1 wk ahead inc case', '2 wk ahead inc case', '3 wk ahead inc case', '4 wk ahead inc case']
    
    start_time = datetime.datetime.strptime('2020-10-05', '%Y-%m-%d').date()
    x = list()
    for index in range(52):
        x.append(start_time)
        start_time = start_time + datetime.timedelta(days= 7)
        sublabel = labels[index][labels[index]['target'] == '4 wk ahead inc case']
        sub_self = self[index][self[index]['target'] == '4 wk ahead inc case']
        wis_label = np.sum(sublabel['WIS'].values)
        self_wis = np.sum(sub_self['WIS'].values)
        L.append(wis_label)
        S.append(self_wis)
    import matplotlib.pyplot as plt
    
    
    plt.plot(x, L, label='CovidHub Ensemble', color='black',marker='o', markersize = 3.0, )
    plt.plot(x, S, label='Our Method', color='red',marker='D', markersize = 2.0)
    plt.title('WIS Comparison for 4th week forecast')
    plt.ylabel('Cumulative WIS')
    plt.xlabel('Time')
    plt.legend()
    plt.show()
    '''