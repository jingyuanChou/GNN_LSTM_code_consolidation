import matplotlib.pyplot as plt
import numpy as np
from idtxl.multivariate_te import MultivariateTE
from idtxl.bivariate_te import BivariateTE
from idtxl.data import Data
from idtxl.visualise_graph import plot_network
import pandas as pd
import pickle
import datetime
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
    starting_time = 60
    time = 119

    print('=================================== Running '+str(time)+'-th timestamp, total 123 timestamps')
    # Arrange the data in a 2D array
    data = hopsitalization[0:time,:]
    data = data.T

    # Convert this into an IDTxl Data object
    data_idtxl = Data(data, dim_order='ps') # use readily available data to calculate transfer entropy

    # Define the adjacency matrix
    # Adjacency matrix should be a 5x5 matrix for 5 nodes (states)
    # with entries being 1 if the corresponding nodes are connected and 0 otherwise.

    # Initialize the MultivariateTE analysis object
    network_analysis = BivariateTE()

    # We should be able to check multiple timestamps, and record the transfer entropy of each [i,j] pair
    # 123 weeks
    # use 0-119, 120-123
    # Set some parameters
    settings = {'cmi_estimator': 'JidtKraskovCMI',
                'max_lag_sources': 2,
                'min_lag_sources': 2,
                'verbose':False}

    # Run the analysis
    results = network_analysis.analyse_network(settings=settings, data=data_idtxl)
    print('ok')
    '''
    # record results
    # the basic idea is we record each pair of source-target, and record the value of their transfer entropy, and draw a graph
    for i in range(0, 5):
        for j in range(0, 5):
            if i == j:
                continue
            else:
                if adjacency[i, j] == 1:
                    temp = results.get_single_target(target = j)
                    sources_of_j = results.get_target_sources(target=j)
                    if i not in sources_of_j:
                        source_target_te[i][j].append(0)
                    else:
                        index = np.where(sources_of_j == i)[0]
                        te = temp['selected_sources_te'][index]
                        source_target_te[i][j].append(te[0])
    '''
    # Plot the network
    #plot_network(results=results, weights='max_te_lag')
    #plt.show()

    # print('Check images')
    '''
    with open('Transfer_entropy_across_time.pkl', 'wb') as f:
        pickle.dump(source_target_te, f)
    
    with open('Transfer_entropy_across_time.pkl', 'rb') as f:
        Transfer_entropy_across_time = pickle.load(f)
    state_map = {'VA':0, 'NC':1, 'SC':2, 'GA':3, 'TN':4}
    # VA: NC, TN
    # NC: VA, SC, TN, GA
    # SC: NC, GA
    # GA: NC, SC, TN
    # TN: GA, VA, NC
    src = 'GA'
    tgt = 'TN'
    src_idx = state_map[src]
    tgt_idx = state_map[tgt]

    fig = plt.figure(figsize=(13, 8))

    ax1 = fig.add_subplot(211)

    start_date = datetime.datetime(2020, 7, 12)
    start_date_float = start_date.timestamp()
    x = [datetime.datetime.fromtimestamp(start_date_float + i * 604800) for i in range(len(hopsitalization))]

    source_target_TF = Transfer_entropy_across_time[src_idx][tgt_idx]
    start_date = datetime.datetime(2020, 7, 12)
    start_date_float = start_date.timestamp()
    Y_start_date = start_date_float + 20 * 604800
    x2 = [datetime.datetime.fromtimestamp(Y_start_date + i * 604800) for i in range(len(hopsitalization) - 20)]

    ax1.plot(x, hopsitalization[src].values,
             label='hospitalizations of {}'.format(src), color='black',
             linestyle='--', marker='o',
             markersize=4.0)
    ax1.plot(x, hopsitalization[tgt].values,
             label='hospitalizations of {}'.format(tgt), color='blue',
             linestyle='--', marker='o',
             markersize=4.0)
    ax1b = ax1.twinx()
    ax1b.plot(x2, source_target_TF,
              label='Transfer Entropy', color='red',
              linestyle='--', marker='o',
              markersize=4.0)
    ax1b.set_ylim(0, 1.5)
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1b.get_legend_handles_labels()
    ax1b.legend(lines + lines2, labels + labels2, loc='upper right')
    plt.title('{}_{}_with_{} laging 2'.format(src, tgt, src))

    #########PLOT FOR SECOND FIGURE
    ax2 = fig.add_subplot(212)

    start_date = datetime.datetime(2020, 7, 12)
    start_date_float = start_date.timestamp()
    x = [datetime.datetime.fromtimestamp(start_date_float + i * 604800) for i in range(len(hopsitalization))] # 604800 is seconds of a week

    source_target_TF = Transfer_entropy_across_time[tgt_idx][src_idx]
    start_date = datetime.datetime(2020, 7, 12)
    start_date_float = start_date.timestamp()
    Y_start_date = start_date_float + 20 * 604800
    x2 = [datetime.datetime.fromtimestamp(Y_start_date + i * 604800) for i in range(len(hopsitalization) - 20)]

    ax2.plot(x, hopsitalization[src].values,
             label='hospitalizations of {}'.format(src), color='black',
             linestyle='--', marker='o',
             markersize=4.0)
    ax2.plot(x, hopsitalization[tgt].values,
             label='hospitalizations of {}'.format(tgt), color='blue',
             linestyle='--', marker='o',
             markersize=4.0)
    ax2b = ax2.twinx()
    ax2b.set_ylim(0, 1.5)
    ax2b.plot(x2, source_target_TF,
              label='Transfer Entropy', color='red',
              linestyle='--', marker='o',
              markersize=4.0)
    lines, labels = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2b.get_legend_handles_labels()
    ax2b.legend(lines + lines2, labels + labels2, loc='upper right')
    plt.title('{}_{} with {} laging 2'.format(tgt, src, tgt))

    plt.show()
    '''