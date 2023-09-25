from torch_geometric.data import InMemoryDataset
import pandas as pd
import json
import datetime
from tqdm import tqdm
from util import *

import matplotlib.pyplot as plt

import numpy as np

import countryinfo
from parameters import Parameters
from math import isnan
from matplotlib import pyplot

params = Parameters()
# Test Recurrent Neural Networks on COVID Dataset
# There is a separate file because the training pattern is slightly different,
# and I am almost exclusively using RNNs at this point.

# Calculate the mean number of new cases for each country for use in the MASE loss function
country_means = None
country_populations = None


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
        fb_mobility = pd.read_csv('/Users/bijiehao/Downloads/causal_gnn_KDD/data/covid-data/facebook_mobility.csv')

        def get_sci(country1, country2):
            DEFAULT = 50000

            def get_country_code(country):
                try:
                    return countryinfo.CountryInfo(country).iso(2)
                except KeyError:
                    return None

            code1 = get_country_code(country1)
            code2 = get_country_code(country2)
            if code1 is None or code2 is None:
                print("DEFAULT")
                return DEFAULT

            row = fb_mobility.loc[fb_mobility.user_loc == code1].loc[
                fb_mobility.fr_loc == code2]
            if row.shape[0] == 0:
                print("DEFAULT")
                return DEFAULT  # The mobility dataset is missing one of the given countries
            else:
                sci = row.iloc[0].scaled_sci
                if isnan(sci):
                    print("DEFAULT")
                    return DEFAULT
                else:
                    return row.iloc[0].scaled_sci

        # Load mobility dataset
        mobility = pd.read_csv('data/covid-data/mobility_data.csv')
        mobility = mobility.loc[mobility.year == 2016]
        '''
        def get_mobility_score(country1, country2):
            DEFAULT = 50000
            if country1 not in nations:
                return DEFAULT
            if country2 not in nations:
                return DEFAULT
            def get_country_name(country):
                if country not in nations:
                    return "Not Found"
                return country
            row = mobility.loc[mobility.source_name == get_country_name(country1)].loc[mobility.target_name == get_country_name(country2)]
            if row.shape[0] == 0:
                return DEFAULT  # The mobility dataset is missing one of the given countries
            else:
                trips = row.iloc[0].estimated_trips
                if isnan(trips):
                    return DEFAULT
                else:
                    return row.iloc[0].estimated_trips
        '''
        # Determine edge_index: Closest 3 countries are connected

        data_list = []
        source_nodes = []
        target_nodes = []
        edge_attrs = []
        '''
        for i in range(n):
            c1 = country_centroids[nations[i]]
            distances = []
            countries = []
            for j in range(n):
                if i == j:  # Don't want self-loops in the graph
                    continue
                c2 = country_centroids[nations[j]]
                dist = geodesic(c1, c2)
                index = bisect.bisect(distances, dist)
                if index < params.EDGES_PER_NODE:
                    distances.insert(index, dist.km)
                    countries.insert(index, j)
                    # if distances[len(distances) - 1] > params.DISTANCE_THRESHOLD:
                    #     distances.pop()
                    #     countries.pop() #Uncomment to create edge between all countries within a distance threshold, or at least a minimum of EDGES_PER_NODE nearest countries
            source_nodes += [i]*params.EDGES_PER_NODE
            target_nodes += countries[:params.EDGES_PER_NODE]
            edge_attrs += distances[:params.EDGES_PER_NODE]
        '''

        col_names = list(hopsitalization.columns)

        for index, state in enumerate(col_names):
            sub_state = neighbouring_states[neighbouring_states['StateCode'] == state]
            target_states = sub_state['NeighborStateCode'].values

            for t in target_states:
                source_nodes.append(index)
                target_nodes.append(col_names.index(t))

        edge_attr = list()

        for i in range(len(source_nodes)):
            edge_attr.append([torch.tensor(1)])
        edge_attr = torch.tensor(edge_attr)
        # Add the mobility feature to the edge weights
        '''
        if len(params.mobility_edge_features) or True:
            for i in range(len(source_nodes)):
                edge_attrs[i] = [edge_attrs[i]]
                if "sci" in params.mobility_edge_features:
                    edge_attrs[i].append(get_sci(nations[source_nodes[i]], nations[target_nodes[i]]))
                if "flights" in params.mobility_edge_features:
                    edge_attrs[i].append(get_mobility_score(nations[source_nodes[i]], nations[target_nodes[i]]))
                if "distance" not in params.mobility_edge_features:
                    edge_attrs[i].pop(0)
        '''
        #
        torch_def = torch.cuda if torch.cuda.is_available() else torch

        node_mask = torch.ones(len(hopsitalization)).bool()
        edge_mask = torch.ones(len(source_nodes)).bool()

        params.edge_count = len(source_nodes)

        # The shape of the dataframe is [2, 48, 335] where dimensions are [feature, nation, date]

        for i in tqdm(range(len(hopsitalization) - params.lookback_pattern[0])):

            '''
            # !Masking currently not being used!
            edge_mask = torch.logical_not(torch.logical_xor(edge_mask, torch.bernoulli(0.95 * torch.ones(len(source_nodes))).bool()))
            node_mask = torch.logical_not(torch.logical_xor(node_mask, torch.bernoulli(0.95 * torch.ones(len(df))).bool()))
            inv_node_mask = ~node_mask
            nodes_to_drop = set(torch.arange(len(df))[inv_node_mask].tolist())
            # dropped nodes
            temp_edge_mask = edge_mask.clone()
            for j in range(len(source_nodes)):
                if source_nodes[j] in nodes_to_drop or target_nodes[j] in nodes_to_drop:
                    temp_edge_mask[j] = False
            '''

            # Node Features
            # mean cases for 37 countries
            values_x = []
            for n in params.lookback_pattern:
                m = i + params.lookback_pattern[0] - n
                temp = [np.asarray([hopsitalization.iloc[m, j]], dtype='float64') for j in range(n_states)]
                values_x.append(np.asarray(temp, dtype='float64'))
            values_x = np.asarray(values_x)
            x = torch_def.FloatTensor(values_x)
            # x = x[node_mask, :]

            # Labels
            values_y = hopsitalization.iloc[(i + params.lookback_pattern[0] + 1):(i + params.lookback_pattern[0] + 5),
                       :].to_numpy().T
            values_y = np.asarray(values_y, dtype='float64')
            y = torch_def.FloatTensor(values_y)
            # y = y[node_mask, :]

            if y.shape[1] != 4:
                break

            # Edge Index
            edge_index = torch_def.LongTensor([source_nodes.copy(), target_nodes.copy()])
            # edge_index = edge_index[:, temp_edge_mask]

            data = Data(x=x, edge_index=edge_index, y=y, edge_attr=edge_attr)
            data_list.append(data)
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


def plot_forecasts(series, forecasts, n_test):
    # plot the entire dataset in blue
    pyplot.plot(series.values)
    # plot the forecasts in red
    for i in range(len(forecasts)):
        off_s = len(series) - n_test + i - 1
        off_e = off_s + len(forecasts[i]) + 1
        xaxis = [x for x in range(off_s, off_e)]
        yaxis = [series.values[off_s]] + forecasts[i]
        pyplot.plot(xaxis, yaxis, color='red')
    # show the plot
    pyplot.show()


def train_on_dataset(train_dataset, val_dataset, test_dataset, visualize=True, record=True):
    """record: Bool — record results in .json file"""

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Set loss function
    loss_func = params.loss_func  # mase2_loss in parameters
    reporting_metric = params.reporting_metric  # mase1_loss in parameters

    models = params.models

    # Setup for results
    results = {
        "Description": params.experiment_description,
        "Models": {},
    }
    train_losseses, val_losseses, test_losseses = [], [], []
    train_rmses, val_rmses, test_rmses = [], [], []

    # For each model...
    for i in range(len(models)):
        model = models[i]
        print(model)

        # Setup for results
        train_losses, val_losses, test_losses = [], [], []
        train_rms, val_rms, test_rms = [], [], []
        results["Models"][model.name] = {
            "Architecture": str(model),
            "Loss by Epoch": [],
            "Reporting Metric by Epoch": [],
            "Loss by States": {'train': {}, 'val': {}, 'test': {}},
            "Test Predictions": [],
            "Test Labels": []
        }

        def forward(snapshot, h, c):
            """
            Deals with slight differences in forward calls between models
            """
            if model.name == 'Lag':
                return model(snapshot, h, c)

            if h is None:
                out = model(snapshot, h, c)
            elif c is None:
                # out = model(snapshot, h.detach(), c)
                out = model(snapshot, h, c)
            else:
                # out = model(snapshot, h.detach(), c.detach())
                out = model(snapshot, h, c)
            if len(out) == 3:
                x, h, c = out
                # h = h.detach()
                return x, h, c
            else:
                x, h = out
                # h = h.detach()
                return x, h

        # For each epoch...
        num_epochs = params.num_epochs

        # Lag model does not optimize
        if model.name != "Lag":
            optimizer = params.get_optimizer(model.parameters())
        else:
            num_epochs = 1

        for epoch in range(num_epochs):

            # Setup for results
            predictions = {'train': [], 'val': [], 'test': []}
            labels = {'train': [], 'val': [], 'test': []}

            # TRAIN MODEL
            model.train()
            train_cost = 0
            train_rm = 0
            # For each training example...
            for time, snapshot in enumerate(train_dataset):
                # Reset cell and hidden states
                h, c = None, None

                # For each snapshot in the example lookback
                for sub_time in range(len(params.lookback_pattern)):
                    # Get output and new cell/hidden states for prediction on example
                    sub_snapshot = Data(x=snapshot.x[sub_time], edge_index=snapshot.edge_index,
                                        edge_attr=snapshot.edge_attr)
                    y_hat, h, c = forward(sub_snapshot, h, c)

                # Calculate the loss from the final prediction of the sequence
                train_cost += reporting_metric(y_hat, snapshot.y)
                train_rm += reporting_metric(y_hat, snapshot.y)

            # Take average of loss from all training examples
            train_rm /= time + 1
            train_cost /= time + 1
            train_rms.append(train_rm)
            train_losses.append(train_cost)  # Keep list of training loss from each epoch

            # Backpropagate, unless lag model
            if model.name != "Lag":
                train_cost.backward()
                optimizer.step()
                optimizer.zero_grad()

            # Evaluate perforamance on train/val/test datasets
            with torch.no_grad():
                model.eval()

                # EVALUATE MODEL - TRAINING
                train_eval_cost = 0
                train_eval_rm = 0
                for time, snapshot in enumerate(train_dataset):
                    h, c = None, None
                    for sub_time in range(len(params.lookback_pattern)):
                        sub_snapshot = Data(x=snapshot.x[sub_time], edge_index=snapshot.edge_index,
                                            edge_attr=snapshot.edge_attr)
                        y_hat, h, c = forward(sub_snapshot, h, c)

                    # Keep a list of the predictions and labels across the entire epoch
                    predictions['train'].append(y_hat)
                    labels['train'].append(snapshot.y)

                    train_eval_rm += reporting_metric(y_hat, snapshot.y)
                    train_eval_cost += reporting_metric(y_hat, snapshot.y)

                train_eval_rm /= time + 1
                train_eval_cost /= time + 1

                # EVALUATE MODEL - VALIDATION
                val_cost = 0
                val_rm = 0
                for time, snapshot in enumerate(val_dataset):
                    h, c = None, None
                    for sub_time in range(len(params.lookback_pattern)):
                        sub_snapshot = Data(x=snapshot.x[sub_time], edge_index=snapshot.edge_index,
                                            edge_attr=snapshot.edge_attr)
                        y_hat, h, c = forward(sub_snapshot, h, c)

                    predictions['val'].append(y_hat)
                    labels['val'].append(snapshot.y)

                    val_rm += reporting_metric(y_hat, snapshot.y)
                    val_cost += reporting_metric(y_hat, snapshot.y)

                val_rm /= time + 1
                val_cost /= time + 1
                val_cost = val_cost.item()
                val_rms.append(val_rm)
                val_losses.append(val_cost)

                # EVALUATE MODEL - TEST
                test_cost = 0
                test_rm = 0
                for time, snapshot in enumerate(test_dataset):
                    h, c = None, None
                    for sub_time in range(len(params.lookback_pattern)):
                        sub_snapshot = Data(x=snapshot.x[sub_time], edge_index=snapshot.edge_index,
                                            edge_attr=snapshot.edge_attr)
                        y_hat, h, c = forward(sub_snapshot, h, c)

                    predictions['test'].append(y_hat)
                    labels['test'].append(snapshot.y)

                    test_rm += reporting_metric(y_hat, snapshot.y)
                    test_cost += reporting_metric(y_hat, snapshot.y)

                test_rm /= time + 1
                test_cost /= time + 1
                test_cost = test_cost.item()
                test_rms.append(test_rm)
                test_losses.append(test_cost)

            # Save to results and display losses for this epoch
            results["Models"][model.name]["Loss by Epoch"].append({
                "Train": float(train_cost),
                "Validation": float(val_cost),
                "Test": float(test_cost)
            })

            results["Models"][model.name]["Reporting Metric by Epoch"].append({
                "Train": float(train_cost),
                "Validation": float(val_rm),
                "Test": float(test_rm)
            })
            tps = predictions['test']
            results["Models"][model.name]['Test Predictions'].append([tp.reshape(-1).tolist() for tp in tps])
            tls = labels['test']
            results["Models"][model.name]['Test Labels'].append([tl.reshape(-1).tolist() for tl in tls])
            print(
                'Epoch: {:03d}, Train Loss: {:.5f}, Val Loss: {:.5f}, Test Loss: {:.5f}'.format(
                    epoch,
                    float(train_cost),
                    val_cost,
                    test_cost))
            print('Epoch: {:03d}, Train RM: {:.5f}, Val RM: {:.5f}, Test RM: {:.5f}'.format(epoch,
                                                                                            float(train_cost),
                                                                                            val_rm,
                                                                                            test_rm))
        # Keep a list of losses from each epoch for every model
        train_losseses.append(train_losses)
        val_losseses.append(val_losses)
        test_losseses.append(test_losses)

        train_rmses.append(train_rms)
        val_rmses.append(val_rms)
        test_rmses.append(test_rms)

        best_epoch = val_losses.index(min(val_losses))
        best_losses = [float(train_losses[best_epoch]), val_losses[best_epoch], test_losses[best_epoch]]
        best_rms = [float(train_rms[best_epoch]), float(val_rms[best_epoch]),
                    float(test_rms[best_epoch])]
        print(
            "BEST EPOCH----" + 'Epoch: {:03d}, Train Loss: {:.5f}, Val Loss: {:.5f}, Test Loss: {:.5f}'.format(
                best_epoch,
                best_losses[0],
                best_losses[1],
                best_losses[2]))
        print(
            "BEST EPOCH----" + 'Epoch: {:03d}, Train RM: {:.5f}, Val RM: {:.5f}, Test RM: {:.5f}'.format(
                best_epoch,
                best_rms[0],
                best_rms[1],
                best_rms[2]))

        # Calculate and save loss per country to results. Optionally, visualize data
        if visualize:
            show_predictions(predictions, labels)

        states = list(hopsitalization.columns)
        '''
        results["Models"][model.name]['Loss by States']['train'] = show_loss_by_states(predictions['train'],
                                                                                       labels['train'], states,
                                                                                       plot=True)
        results["Models"][model.name]['Loss by States']['val'] = show_loss_by_states(predictions['val'],
                                                                                     labels['val'], states,
                                                                                     plot=True)
        results["Models"][model.name]['Loss by States']['test'] = show_loss_by_states(predictions['test'],
                                                                                      labels['test'], states,
                                                                                      plot=True)
        '''
        results["Models"][model.name]['Loss by States']['all_dataset'] = show_predictions_labels_by_states(predictions,
                                                                                                           labels,
                                                                                                           states,
                                                                                                           plot=True)

        results["Models"][model.name]['best_epoch'] = {
            "Loss": best_losses,
            "Reporting Metric": best_rms,
            "Test Predictions": results["Models"][model.name]['Test Predictions'][best_epoch],
            "Test Labels": results["Models"][model.name]['Test Labels'][best_epoch]
        }

        # show_labels_by_country(labels, nations)

    if visualize:
        # Set labels and plot loss curves for validation
        x = np.arange(0, num_epochs)
        plt.title('Model Comparison')
        plt.xlabel('Epoch')
        plt.ylabel('MAE Loss')
        for i in range(len(models)):
            label = models[i].name
            plt.plot(x, train_losseses[i], label=str(label) + " (train)")
            plt.plot(x, val_losseses[i], label=str(label) + " (val)")
            plt.plot(x, test_losseses[i], label=str(label) + " (test)")
        plt.legend()
        plt.show()

    if record:
        # Save results into a .json file
        date = datetime.datetime.now().isoformat().split(".")[0]
        with open(f'results/4_weeks_results.json', 'w') as f:
            json.dump(results, f, indent=4)

    return results


def gnn_predictor():
    # Load, shuffle, and split dataset
    dataset = COVIDDatasetSpaced(root='data/covid-data/')

    sample = len(dataset)
    sample *= params.sample  # Optionally, choose a frame of the dataset to work with
    train_dataset = dataset[:int(0.8 * sample)]
    val_dataset = dataset[int(0.8 * sample):int(0.9 * sample)]
    test_dataset = dataset[int(0.9 * sample):int(sample)]

    train_on_dataset(train_dataset, val_dataset, test_dataset, visualize=True, record=True)


def cross_validate():
    # Load, shuffle, and split dataset
    dataset = COVIDDatasetSpaced(root='data/covid-data/')
    dataset = dataset.shuffle()

    sample = len(dataset)
    sample *= params.sample  # Optionally, choose a frame of the dataset to work with

    train_datasets = [
        dataset[:int(0.8 * i / params.K * sample)] + dataset[int(0.8 * (i + 1) / params.K * sample):int(0.8 * sample)]
        for i in range(params.K)]
    val_datasets = [dataset[int(0.8 * i / params.K * sample): int(0.8 * (i + 1) / params.K * sample)] for i in
                    range(params.K)]
    test_dataset = dataset[int(0.8 * sample):int(sample)]

    best_results = []
    lowest_val = float('inf')
    for i in range(params.K):
        results = train_on_dataset(train_datasets[i], val_datasets[i], test_dataset, visualize=True, record=False)

        if results['best_epoch'][2] < lowest_val:
            lowest_val = results['best_epoch'][2]
            best_results = results['best_epoch']
    print("BEST RESULTS: ", best_results)


if __name__ == '__main__':
    # Get country centroids data
    df2 = pd.read_csv("country_centroids.csv")
    states = pd.read_csv('/Users/bijiehao/Downloads/causal_gnn_KDD/covid-data/state_hhs_map.csv')
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
    print('true')

    # '2020-07-12' is the first date that has data, and we will start from here

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
    n_states = len(states_code)
    states_map = states_code.set_index('ABB').T.to_dict('list')
    states_fips_map_index = states_code.set_index('FIPS').T.to_dict('list')

    source_nodes = []
    target_nodes = []

    col = hopsitalization.columns
    new_col = list()
    for fips in col:
        new_col.extend(states_fips_map_index[int(fips)])

    hopsitalization.columns = new_col

    '''
    df = pd.read_csv('/Users/bijiehao/Downloads/GNND-master/covid-data/covid-19-world-cases-deaths-testing.csv')

    columns = ['location', 'date'] + params.features
    df = df.filter(columns)
    df = df.fillna(method="ffill").fillna(method="bfill").fillna(0)

    df = df[df.location.isin(df2.name_long.values)]
    nations = df.location.unique()
    remove_me = ['Faeroe Islands', 'Isle of Man', 'Jersey', 'Guernsey', 'Vatican', 'Kosovo', 'Monaco', 'San Marino', 'Liechtenstein', 'Andorra']
    for n in remove_me:
        nations = nations[nations != n]

    # Clean the dataset for negative values in the new_cases column - convert them to the average of the surrounding 7 days
    if 'new_cases' in df:
        new_cases = df.new_cases.tolist()
        for i in range(len(new_cases)):
            if new_cases[i] < 0:
                new_cases[i] = sum(new_cases[i-3:i+4])/7
        df.new_cases = new_cases

    # Only needed if remaking the dataset
    if not os.path.exists('data/covid-data/processed/covid_dataset_spaced.dataset') or True:
        dates = sorted(df.date.unique())
        new_data = []
        for j in range(len(params.features)):
            new_data.append({'Time': range(len(dates))})

        print("Pre-Processing Data")
        for i in tqdm(range(len(nations))):
            nation = nations[i]
            nation_data = df.loc[df.location == nation]
            new_features = [[] for i in range(len(params.features))]
            last_values = [0.0] * len(params.features)
            for date in dates:
                date_row = nation_data.loc[nation_data.date == date]
                if not date_row.empty:
                    for j in range(len(params.features)):
                        new_features[j].append(date_row[params.features[j]].iloc[0])
                        last_values[j] = date_row.iloc[0][params.features[j]]
                else:
                    for j in range(len(params.features)):
                        new_features[j].append(last_values[j])
            for j in range(len(params.features)):
                new_data[j][nation] = new_features[j]

        dfs = [pd.DataFrame(data=new_data[j]) for j in range(len(params.features))]
        df = dfs[0]

        print('Dataset preprocessed')
        df.to_csv("df.csv")
        print(df.head())
        print(df.columns)
        print(df.shape)

        country_means = [0]*(df.shape[1]-1)
        for i in range(df.shape[0]):
            for j in range(1, df.shape[1]):
                country_means[j-1] += df.iloc[i][j] # sum of cases of countries
        for i in range(len(country_means)):
            country_means[i] = country_means[i] / df.shape[0] # average cases per day per country
            if country_means[i] == 0:
                country_means[i] = 0.01

        country_means = torch.FloatTensor(country_means)

    #Get centroid of each country
    country_centroids = {}
    for nation in nations:
        match = df2.loc[df2.name_long == nation]
        if len(match):
            lon = match.Longitude.values[0]
            lat = match.Latitude.values[0]
            country_centroids[nation] = (lat, lon)
            # print(nation + ": https://www.google.com/maps/place/" + str(lat) + "," + str(lon))
        else:
            print("Missing coordinates for country", nation)

    # Get population of each country
        
    for nation in nations:
        try:
            country_populations.append(countryinfo.CountryInfo(nation).population())
        except KeyError:
            country_populations.append(0)
    '''
    if params.CROSS_VALIDATE:
        cross_validate()
    else:
        gnn_predictor()
