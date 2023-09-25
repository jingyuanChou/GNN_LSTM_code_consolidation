from torch_geometric.data import InMemoryDataset
import pandas as pd
from tqdm import tqdm
from util import *
import matplotlib.pyplot as plt
import numpy as np
import countryinfo
from parameters import Parameters
from math import isnan

params = Parameters()


# Test Recurrent Neural Networks on COVID Dataset
# There is a separate file because the training pattern is slightly different,
# and I am almost exclusively using RNNs at this point.


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

        data_list = []
        source_nodes = []
        target_nodes = []
        edge_attrs = []
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

        torch_def = torch.cuda if torch.cuda.is_available() else torch

        node_mask = torch.ones(len(hopsitalization)).bool()
        edge_mask = torch.ones(len(source_nodes)).bool()

        params.edge_count = len(source_nodes)

        # The shape of the dataframe is [2, 48, 335] where dimensions are [feature, nation, date]

        for i in tqdm(range(len(hopsitalization) - params.lookback_pattern[0])):

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
            values_y = hopsitalization.iloc[(i + params.lookback_pattern[0] + 1):(i + params.lookback_pattern[0] + 2),
                       :].to_numpy().T
            values_y = np.asarray(values_y, dtype='float64')
            y = torch_def.FloatTensor(values_y)
            # y = y[node_mask, :]

            if y.shape[1] != 1:
                break

            # Edge Index
            edge_index = torch_def.LongTensor([source_nodes.copy(), target_nodes.copy()])
            # edge_index = edge_index[:, temp_edge_mask]

            data = Data(x=x, edge_index=edge_index, y=y, edge_attr=edge_attr)
            data_list.append(data)
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


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

    # For each model...
    for i in range(len(models)):
        model = models[i]
        # Setup for results
        print(model)
        train_losses, val_losses, test_losses = [], [], []
        results["Models"][model.name] = {
            "Architecture": str(model),
            "Loss by Epoch": [],
            'train_loss': [],
            'val_loss': [],
            "Reporting Metric by Epoch": [],
        }

        # For each epoch...
        num_epochs = params.num_epochs
        # Lag model does not optimize
        if model.name != "Lag":
            optimizer = params.get_optimizer(model.parameters())
        else:
            num_epochs = 1
        # minimum_loss = sys.maxsize
        for epoch in range(num_epochs):
            # Setup for results
            predictions = {'train': [], 'val': [], 'test': []}
            labels = {'train': [], 'val': [], 'test': []}
            # TRAIN MODEL
            model.train()
            train_cost = 0
            # For each training example...
            h, c = None, None
            h_list, c_list = list(), list()
            for time, snapshot in enumerate(train_dataset):
                # snapshot,
                for sub_time in range(len(params.lookback_pattern)):
                    # Get output and new cell/hidden states for prediction on example
                    sub_snapshot = Data(x=snapshot.x[sub_time], edge_index=snapshot.edge_index,
                                        edge_attr=snapshot.edge_attr)
                    y_hat, h, c = model(sub_snapshot, h, c)

                    h_list.append(h)
                    c_list.append(c)
                # Calculate the loss from the final prediction of the sequence
                # y_hat = y_hat[0]
                train_cost += reporting_metric(y_hat, snapshot.y)

            # Take average of loss from all training examples
            train_cost /= time + 1
            train_losses.append(train_cost)  # Keep list of training loss from each epoch

            # Backpropagate, unless lag model
            if model.name != "Lag":
                train_cost.backward()
                optimizer.step()
                optimizer.zero_grad()


            # Evaluate perforamance on train/val
            '''
            with torch.no_grad():
                model.eval()

                # EVALUATE MODEL - VALIDATION
                val_cost = 0
                for time, snapshot in enumerate(val_dataset):
                    for sub_time in range(len(params.lookback_pattern)):
                        sub_snapshot = Data(x=snapshot.x[sub_time], edge_index=snapshot.edge_index,
                                            edge_attr=snapshot.edge_attr)
                        y_hat, h, c = model(sub_snapshot, h, c)

                    predictions['val'].append(y_hat)
                    labels['val'].append(snapshot.y)
                    # y_hat = y_hat[0]
                    val_cost += reporting_metric(y_hat, snapshot.y)

                val_cost /= time + 1
                val_cost = val_cost.item()
                val_losses.append(val_cost)
            # Save to results and display losses for this epoch
            results["Models"][model.name]["Loss by Epoch"].append({
                "Train": float(train_cost),
                "Validation": float(val_cost)
            })
            '''

            print('Epoch: {:03d}, Train Loss: {:.5f}'.format(epoch, float(train_cost)))
        # Keep a list of losses from each epoch for every model

        train_losseses.append(train_losses)
        val_losseses.append(val_losses)
        # test_losseses.append(test_losses)

        results["Models"][model.name]['train_loss'] = train_losseses
        results["Models"][model.name]['val_loss'] = val_losseses
        # results["Models"][model.name]['test_loss'] = test_losseses

        best_epoch = val_losses.index(min(val_losses))
        best_losses = [float(train_losses[best_epoch]), val_losses[best_epoch]]
        print(
            "BEST EPOCH----" + 'Epoch: {:03d}, Train Loss: {:.5f}, Val Loss: {:.5f}'.format(
                best_epoch,
                best_losses[0],
                best_losses[1]))

        # loss plot against epochs

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
                # plt.plot(x, test_losseses[i], label=str(label) + "(test)")
            plt.legend()
            plt.show()

        candidate_states = ['NY', 'CA', 'TX', 'VA']
        states = list(hopsitalization.columns)

        pred_ca = list()
        temp_y = list()
        final_looping_prediction = list()

        # looping prediction for training set
        for index, snapshot in enumerate(train_dataset):
            final_snapshot = snapshot.x
            test_edge_index = snapshot.edge_index
            test_edge_attr = snapshot.edge_attr
            temp = list()
            for i in range(4):
                for sub_time in range(len(params.lookback_pattern)):
                    sub_snapshot = Data(x=final_snapshot[sub_time], edge_index=test_edge_index,
                                        edge_attr=test_edge_attr)
                    y_hat, h, c = model(sub_snapshot, h, c)
                    temp.append(y_hat)
                pred_ca.append(y_hat)
                y_hat = torch.unsqueeze(y_hat, 0)
                temp_y.append(y_hat)
                nexy_y = torch.unsqueeze(y_hat[:, :, 0], 2)
                final_snapshot = torch.cat((final_snapshot[1:], nexy_y), dim=0)
            final_looping_prediction.append(pred_ca)
            pred_ca = list()
        for can in candidate_states:
            can_idx = states.index(can)
            x = np.arange(0, len(train_dataset))
            train_label = [tra.y for tra in train_dataset]
            label_train = [train_label[time][can_idx] for time in range(len(train_label))]
            expected_train_labels = list()
            for ele in label_train:
                candidate = ele[0]
                expected_train_labels.append(candidate)
            plt.plot(x, expected_train_labels, label='{} labels'.format(can), color='black')
            start = 0
            while start < len(final_looping_prediction):
                xaxis = x[start: start + 4]
                if len(xaxis) != 4:
                    break

                next_4_pred = list()
                for each_4_time in final_looping_prediction[start]:
                    next_4_pred.append(each_4_time[can_idx][0])

                if start == 0:
                    plt.plot(xaxis, next_4_pred, label='{} predictions'.format(can), color='blue', linestyle='dashdot')
                else:
                    plt.plot(xaxis, next_4_pred, color='blue', linestyle='dashdot')
                start = start + 1

            plt.title('{} testing set prediction'.format(can))
            plt.legend()
            plt.show()



    for can in candidate_states:
        can_idx = states.index(can)
        # 1 week ahead predictions
        pred_ca = list()
        for index, snapshot in enumerate(test_dataset):
            final_snapshot = snapshot.x
            test_edge_index = snapshot.edge_index
            test_edge_attr = snapshot.edge_attr
            temp = list()
            for sub_time in range(len(params.lookback_pattern)):
                sub_snapshot = Data(x=final_snapshot[sub_time], edge_index=test_edge_index,
                                    edge_attr=test_edge_attr)
                y_hat, h, c = model(sub_snapshot, h, c)
                temp.append(y_hat)
            pred_ca.append(y_hat)

        x = np.arange(0, len(test_dataset))
        test_label = [tst.y for tst in test_dataset]
        label_test = [test_label[time][can_idx] for time in range(len(test_label))]
        expected_test_labels = list()
        for ele in label_test:
            candidate = ele[0]
            expected_test_labels.append(candidate)
        plt.plot(x, expected_test_labels, label='{} labels'.format(can), color='black')
        pred_0_temp = list()
        for ca_each in pred_ca:
            next_y = ca_each[can_idx]
            pred_0_temp.append(next_y[0])
        plt.plot(x, pred_0_temp, label='{} predictions'.format(can), color='red', linestyle='dashdot')
        plt.title('{} testing set prediction'.format(can))
        plt.legend()
        plt.show()

        # Plot CA predictions

        pred_ca = list()
        for index, snapshot in enumerate(train_dataset):
            final_snapshot = snapshot.x
            test_edge_index = snapshot.edge_index
            test_edge_attr = snapshot.edge_attr
            for sub_time in range(len(params.lookback_pattern)):
                # Get output and new cell/hidden states for prediction on example
                sub_snapshot = Data(x=final_snapshot[sub_time], edge_index=test_edge_index,
                                    edge_attr=test_edge_attr)
                y_hat, h, c = model(sub_snapshot, h, c)

            pred_ca.append(y_hat)

        x = np.arange(0, len(train_dataset))
        train_label = labels['train']
        label_train = [train_label[time][can_idx] for time in range(len(train_label))]
        expected_labels = list()
        for ele in label_train:
            candidate = ele[0]
            expected_labels.append(candidate)
        plt.plot(x, expected_labels, label='{} labels'.format(can), color='black')
        plt.plot(x, [ca_each[can_idx].item() for ca_each in pred_ca], label='{} predictions'.format(str(can)),
                 color='red', linestyle='dashdot')
        plt.title('{} training set prediction'.format(can))
        plt.legend()
        plt.show()

        # Plot validation set
        pred_ca = list()
        for index, snapshot in enumerate(val_dataset):
            final_snapshot = snapshot.x
            val_edge_index = snapshot.edge_index
            val_edge_attr = snapshot.edge_attr
            for sub_time in range(len(params.lookback_pattern)):
                sub_snapshot = Data(x=final_snapshot[sub_time], edge_index=val_edge_index,
                                    edge_attr=val_edge_attr)
                y_hat, h, c = model(sub_snapshot, h, c)
            pred_ca.append(y_hat)
        x = np.arange(0, len(val_dataset))
        val_label = labels['val']
        label_val = [val_label[time][can_idx] for time in range(len(val_label))]
        expected_labels = list()
        for ele in label_val:
            candidate = ele[0]
            expected_labels.append(candidate)
        plt.plot(x, expected_labels, label='{} labels'.format(can), color='black')
        plt.plot(x, [ca_each[can_idx].item() for ca_each in pred_ca], label='{} predictions'.format(can), color='red',
                 linestyle='dashdot')
        plt.title('{} validation set prediction'.format(can))
        plt.legend()
        plt.show()

        results["Models"][model.name]['best_epoch'] = {
            "Loss": best_losses
        }

        # show_labels_by_country(labels, nations)

    '''
    if record:
        # Save results into a .json file
        date = datetime.datetime.now().isoformat().split(".")[0]
        with open(f'results/4_weeks_results.json', 'w') as f:
            json.dump(results, f, indent=4)
    '''
    return results


def gnn_predictor():
    # Load, shuffle, and split dataset
    dataset = COVIDDatasetSpaced(root='data/covid-data/')

    sample = len(dataset)
    sample *= params.sample  # Optionally, choose a frame of the dataset to work with
    train_dataset = dataset[:int(0.8 * sample)]
    val_dataset = dataset[int(0.8 * sample):int(1.0 * sample)]
    test_dataset = None
    train_on_dataset(train_dataset, val_dataset, test_dataset, visualize=True, record=True)


def cross_validate():
    # Load, shuffle, and split dataset
    dataset = COVIDDatasetSpaced(root='data/covid-data/')
    # dataset = dataset.shuffle()

    sample = len(dataset)

    sample *= params.sample  # Optionally, choose a frame of the dataset to work with
    # 5-FOLD CV
    train_datasets = list()
    '''
    for i in range(params.K):
        print(int(0.8 * i / params.K * sample))
        print(int(0.8 * (i + 1) / params.K * sample))
        print(int(0.8 * sample))
        temp = dataset[:int(0.8 * i / params.K * sample)] + dataset[
                                                            int(0.8 * (i + 1) / params.K * sample):int(0.8 * sample)]
        train_datasets.append(temp)

    val_datasets = [dataset[int(0.8 * i / params.K * sample): int(0.8 * (i + 1) / params.K * sample)] for i in
                    range(params.K)]
    '''
    # TIME-SERIES 4 FOLD CV
    val_datasets = list()
    for i in range(params.K - 1):
        train_datasets.append(dataset[:int(0.8 * (i + 1) / params.K * sample)].shuffle())
        val_datasets.append(dataset[int(0.8 * (i + 1) / params.K * sample): int(0.8 * (i + 2) / params.K * sample)])

    test_dataset = dataset[int(0.8 * sample):int(sample)]

    best_results = []
    lowest_val = float('inf')
    cv_results = list()
    for i in range(params.K - 1):
        results = train_on_dataset(train_datasets[i], val_datasets[i], test_dataset, visualize=True, record=False)
        cv_results.append(results)
        if results['Models']['Our Model']['best_epoch']['Loss'][1] < lowest_val:
            lowest_val = results['Models']['Our Model']['best_epoch']['Loss'][1]
            best_results = results['Models']['Our Model']['best_epoch']
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
    if params.CROSS_VALIDATE:
        cross_validate()
    else:
        gnn_predictor()
