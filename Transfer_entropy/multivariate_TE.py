import matplotlib.pyplot as plt
import numpy as np
from idtxl.multivariate_te import MultivariateTE
from idtxl.data import Data
from idtxl.visualise_graph import plot_network
import pandas as pd
import pickle
import datetime




if __name__ == '__main__':
    states = pd.read_csv('state_hhs_map.csv')
    # VA: 51
    # SC: 45
    # NC: 37
    # GA: 13
    # TN: 47

    selected_states = ['VA', 'SC', 'NC', 'GA', 'TN']
    selected_fips = [51, 45, 37, 13, 47]

    state_fips_dict = {'VA':51,'SC':45, 'NC':37,'GA':13, 'TN':47}
    fips_state_dict = {51:'VA',45:'SC',37:'NC',13:'GA',47:'TN'}

    columns = ['name_long', 'Longitude', 'Latitude', 'continent']
    # Choose only a single continent for smaller testing (can choose any continent)

    hopsitalization = pd.read_csv('hosp_weekly_filt_case_data.csv')
    hopsitalization = hopsitalization.iloc[:-1, :]
    hopsitalization = hopsitalization.T
    hopsitalization.columns = hopsitalization.iloc[0, :]
    hopsitalization = hopsitalization.iloc[1:, :]
    hopsitalization = hopsitalization.iloc[28:, :]

    # neighboring states
    neighbouring_states = pd.read_csv('neighbors-states.csv')
    states_code = pd.read_csv('state_hhs_map.csv',header=None)

    neighbouring_states_copy = neighbouring_states[['NeighborStateCode', 'StateCode']]
    neighbouring_states_copy.columns = ['StateCode', 'NeighborStateCode']

    neighbouring_states = pd.concat((neighbouring_states,neighbouring_states_copy))
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

    VA_data = np.array(hopsitalization['VA'].values)
    NC_data = np.array(hopsitalization['NC'].values)
    SC_data = np.array(hopsitalization['SC'].values)
    GA_data = np.array(hopsitalization['GA'].values)
    TN_data = np.array(hopsitalization['TN'].values)

    VA_data = list(map(int, VA_data))
    NC_data = list(map(int, NC_data))
    SC_data = list(map(int, SC_data))
    GA_data = list(map(int, GA_data))
    TN_data = list(map(int, TN_data))

    # Impose Adjacency matrix to the transfer entropy calculation
    # We can also delete this if not needed.
    adjacency = np.array([[0, 1, 0, 0, 1],  # 0: VA, 1: NC, 2: SC, 3: GA, 4: TN
                          [1, 0, 1, 1, 1],
                          [0, 1, 0, 1, 0],
                          [0, 1, 1, 0, 1],
                          [1, 1, 0, 1, 0]])

    source_target_te = dict()
    for i in range(0,5):
        for j in range(0,5):
            if i == j:
                continue
            else:
                if adjacency[i,j] == 1:
                    if i not in source_target_te.keys():
                        source_target_te[i] = dict()
                        source_target_te[i][j] = list()
                    else:
                        source_target_te[i][j] = list()


    for time in range(20, len(VA_data)):
        print('=================================== Running '+str(time)+'-th timestamp, total 123 timestamps')
        VA_data_sub = VA_data[0:time+1]
        NC_data_sub = NC_data[0:time+1]
        SC_data_sub = SC_data[0:time+1]
        GA_data_sub = GA_data[0:time+1]
        TN_data_sub = TN_data[0:time+1]


        # Arrange the data in a 2D array
        data = np.array([VA_data_sub, NC_data_sub, SC_data_sub, GA_data_sub, TN_data_sub])

        # Convert this into an IDTxl Data object
        data_idtxl = Data(data, dim_order='ps')

        # Define the adjacency matrix
        # Adjacency matrix should be a 5x5 matrix for 5 nodes (states)
        # with entries being 1 if the corresponding nodes are connected and 0 otherwise.

        # Initialize the MultivariateTE analysis object
        network_analysis = MultivariateTE()

        # We should be able to check multiple timestamps, and record the transfer entropy of each [i,j] pair
        adjacency = np.array([[0, 1, 0, 0, 1],  # 0: VA, 1: NC, 2: SC, 3: GA, 4: TN
                              [1, 0, 1, 1, 1],
                              [0, 1, 0, 1, 0],
                              [0, 1, 1, 0, 1],
                              [1, 1, 0, 1, 0]])
        selected_vars = [(i, j) for i in range(5) for j in range(5) if adjacency[i][j] == 1]

        # Set some parameters
        settings = {'cmi_estimator': 'JidtKraskovCMI',
                    'selected_vars_full': selected_vars,
                    'max_lag_sources': 2,
                    'min_lag_sources': 2,
                    'verbose':False}

        # Run the analysis
        results = network_analysis.analyse_network(settings=settings, data=data_idtxl)

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
                            index =  np.where(sources_of_j == i)[0]
                            te = temp['selected_sources_te'][index]
                            source_target_te[i][j].append(te[0])

        # Plot the network
        # plot_network(results=results, weights='max_te_lag')
        # plt.show()

        # print('Check images')
    '''
    with open('Transfer_entropy_across_time.pkl', 'wb') as f:
        pickle.dump(source_target_te, f)
    '''
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