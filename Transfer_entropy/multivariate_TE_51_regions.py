import pickle

import matplotlib.pyplot as plt
import numpy as np
from idtxl.multivariate_te import MultivariateTE
from idtxl.bivariate_te import BivariateTE
from idtxl.data import Data
import pandas as pd
import networkx as nx
import torch




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
    print(fips_states.columns)
    fips_2_state = fips_states.set_index(0)[2].to_dict()
    state_2_fips = fips_states.set_index(2)[0].to_dict()
    state_len = len(fips_list)
    hopsitalization['state'] = hopsitalization['state'].map(fips_2_state)

    index_2_state = hopsitalization.set_index('index')['state'].to_dict()
    state_2_index = hopsitalization.set_index('state')['index'].to_dict()

    neighbouring_states['StateCode'] = neighbouring_states['StateCode'].map(state_2_index)
    neighbouring_states['NeighborStateCode'] = neighbouring_states['NeighborStateCode'].map(state_2_index)

    G = nx.from_pandas_edgelist(neighbouring_states, 'StateCode', 'NeighborStateCode')
    adj = nx.adjacency_matrix(G)
    hopsitalization = hopsitalization.iloc[:, 30:]  # remove all 0 datapoints
    hopsitalization = hopsitalization.T.values

    list_of_next_4_since_current_time = list()
    adj = torch.from_numpy(adj.A)
    time = 119

    with open("index_2_state", "wb") as idx_2_state:  # Pickling
        pickle.dump(index_2_state, idx_2_state)

    # remove alaska and hawaii, thus 0 and 11 index must be removed

    print('=================================== Running '+str(time)+'-th timestamp, total 123 timestamps')
    # Arrange the data in a 2D array
    data = hopsitalization[0:time,:]
    data = data.T
    # Convert this into an IDTxl Data object
    data_idtxl = Data(data, dim_order='ps') # use readily available data to calculate transfer entropy

    # Initialize the MultivariateTE analysis object
    network_analysis = MultivariateTE()

    # We should be able to check multiple timestamps, and record the transfer entropy of each [i,j] pair
    # Set some parameters
    settings = {'cmi_estimator': 'JidtKraskovCMI',
                'max_lag_sources': 2,
                'min_lag_sources': 2,
                'verbose':False}

    source_target_te = dict()
    for i in range(0,49):
        for j in range(0, 49):
            if i == j:
                continue
            else:
                if i not in source_target_te.keys():
                    source_target_te[i] = dict()
                    source_target_te[i][j] = list()
                else:
                    source_target_te[i][j] = list()

    # Run the analysis
    results = network_analysis.analyse_network(settings=settings, data=data_idtxl)

    for i in range(0, 49):
        for j in range(0, 49):
            if i == j:
                continue
            else:
                temp = results.get_single_target(target=j)
                sources_of_j = [temp_Ls[0] for temp_Ls in temp['selected_vars_sources']]
                if i not in sources_of_j:
                    source_target_te[i][j].append(0)
                else:
                    index = sources_of_j.index(i)
                    te = temp['selected_sources_te'][index]
                    source_target_te[i][j].append(te)

    with open('MTE_49_states.pkl', 'wb') as f:
        pickle.dump(source_target_te, f)


