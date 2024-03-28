import networkx as nx
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from torch_geometric.data import InMemoryDataset
from tqdm import tqdm
from rnn import LSTM
from weight_sage import WeightedSAGEConv
from torch_geometric.utils.convert import from_networkx


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


        Pyg_graph = from_networkx(G)
        edge_index = Pyg_graph.edge_index
        edge_attr = list()
        for i in range(edge_index.shape[1]):
            edge_attr.append([torch.tensor(1)])
        edge_attr = torch.tensor(edge_attr)
        torch_def = torch.cuda if torch.cuda.is_available() else torch

        for i in tqdm(range(len(hopsitalization) - lookback_pattern[0])):
            # Node Features
            values_x = []
            for n in lookback_pattern:
                m = i + lookback_pattern[0] - n
                temp = [np.asarray([hopsitalization[m, j]], dtype='float64') for j in range(49)]
                values_x.append(np.asarray(temp, dtype='float64'))
            values_x = np.asarray(values_x)
            x = torch_def.FloatTensor(values_x)
            # Labels
            if (i + lookback_pattern[0] + 1) == len(hopsitalization) - 1:
                break
            else:
                if (i + 12 + 4)> len(hopsitalization):
                    break
                values_y = hopsitalization[(i + lookback_pattern[0] + 1):(i + lookback_pattern[0] + 5),
                           :].T
                values_y = np.asarray(values_y, dtype='float64')
            y = torch_def.FloatTensor(values_y)
            # Edge Index
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
        self.act = torch.nn.ReLU()
        self.l2_penalty = l2_penalty
        if skip_connection:
            self.linear = torch.nn.Linear(hidden_dim * 2, output_dim)
        else:
            self.linear = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, data, h=None, c=None):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        x = self.graphsage(x, edge_index, edge_attr)
        embeddings = x.detach().cpu().numpy()
        x = self.act(x)
        x = self.dropout(x)
        h1, c = self.lstm1(x, edge_index, edge_attr, h, c)
        h1 = self.dropout(h1)
        h1 = self.act(h1)
        h2, c = self.lstm2(x, edge_index, edge_attr, h1, c)
        h2 = self.dropout(h2)
        h2 = self.act(h2)
        if self.skip_connection:
            x = torch.cat((x, h2), dim=2)
        x = self.linear(x[-1, :, :])

        return x, h, c, embeddings


if __name__ == '__main__':

    # Define the number of previous and future time steps to use
    num_prev_time_steps = 12
    num_future_time_steps = 4
    # Define the hidden dimension size
    hidden_dim = 16
    # Define the learning rate and number of epochs to train for
    lr = 0.01
    num_epochs = 10
    dropout_rate = 0.2
    l2_penalty = 0.1
    horizon = 4
    lookback_pattern = [11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0]

    hopsitalization = pd.read_csv('FORECASTING_GNN/hosp_weekly_filt_case_data.csv')
    hopsitalization = hopsitalization.reset_index()
    hopsitalization['state'] = hopsitalization['state'].astype(int)

    fips_list = hopsitalization['state'].values
    fips_2_index = hopsitalization.set_index('state')['index'].to_dict()
    index_2_fips = hopsitalization.set_index('index')['state'].to_dict()

    neighbouring_states = pd.read_csv('covid-data/neighbouring_states.csv')
    states_code = pd.read_csv('covid-data/state_hhs_map.csv')

    neighbouring_states_copy = neighbouring_states[['NeighborStateCode', 'StateCode']]
    neighbouring_states_copy.columns = ['StateCode', 'NeighborStateCode']

    neighbouring_states = neighbouring_states.append(neighbouring_states_copy)
    neighbouring_states.drop_duplicates()

    fips_states = pd.read_csv('covid-data/state_hhs_map.csv', header=None)
    fips_2_state = fips_states.set_index(0)[2].to_dict()
    state_2_fips = fips_states.set_index(2)[0].to_dict()
    hopsitalization['state'] = hopsitalization['state'].map(fips_2_state)

    index_2_state = hopsitalization.set_index('index')['state'].to_dict()
    state_2_index = hopsitalization.set_index('state')['index'].to_dict()

    neighbouring_states['StateCode'] = neighbouring_states['StateCode'].map(state_2_index)
    neighbouring_states['NeighborStateCode'] = neighbouring_states['NeighborStateCode'].map(state_2_index)

    G = nx.from_pandas_edgelist(neighbouring_states, 'StateCode', 'NeighborStateCode')

    hopsitalization = hopsitalization.iloc[:, 30:]  # remove all 0 datapoints
    hopsitalization = hopsitalization.T.values

    dataset = COVIDDatasetSpaced(root='data/covid-data/')
    # train a new model for every new timestamp
    num_samples_dropout = 50
    filepath = 'best_model_weights_24_AAAI_BASELINE.pt'
    all_50_predictions_for_all_timestamps = list()
    best_loss = float('inf')

    train_idx = int(len(dataset) * 0.8)
    val_idx = len(dataset) - 1

    train_dataset = dataset[:train_idx]
    val_dataset = dataset[train_idx:val_idx]
    test_dataset = dataset[val_idx]
    model = GraphLSTMModel(input_dim=1, hidden_dim=hidden_dim, output_dim=1, module=WeightedSAGEConv,
                           skip_connection=True,
                           dropout_rate=dropout_rate, l2_penalty=l2_penalty)
    # Define the loss function and optimizer
    criterion = torch.nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=model.l2_penalty)

    # Train the model
    samples_50 = list()
    loss_ls = list()
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        # Select the previous and future time steps
        h = None
        c = None
        # Forward pass
        loss = 0
        # We take 80% date to be training data, and 20% to be validation data
        data_length = len(train_dataset)
        validation_loss = 0
        for time, snapshot in enumerate(train_dataset):
            y_pred, h, c, embeddings = model(snapshot, h, c)
            y_pred_update = torch.squeeze(y_pred)
            label_transform = torch.squeeze(torch.transpose(snapshot.y, 1, 0))
            # Compute loss
            loss = loss + criterion(y_pred_update, label_transform)
            # Take average of loss from all training examples
        loss /= time + 1
        loss.backward()
        print(loss.item())
        loss_ls.append(loss.item())
        val_loss_ls = list()

        with torch.no_grad():
            model.eval()
            val_loss = 0
            for val_time, val_snapshot in enumerate(val_dataset):
                val_y_pred, h, c, embeddings = model(val_snapshot)
                val_y_pred_update = torch.squeeze(val_y_pred)
                val_label_transform = torch.squeeze(torch.transpose(val_snapshot.y, 1, 0))
                # Compute loss
                val_loss = val_loss + criterion(val_y_pred_update, val_label_transform)
                # Take average of loss from all training examples
            val_loss /= len(val_dataset)

            if val_loss < best_loss:
                best_loss = val_loss
                torch.save(model.state_dict(), filepath)
            print(
                'Epoch: {:03d}, Train Loss: {:.5f}, Val Loss: {:.5f}'.format(
                    epoch,
                    float(loss.item()),
                    val_loss.item()))
            val_loss_ls.append(val_loss)
    # prediction for next 4 timestamps and predict in autoregressive way
    best_model = model.load_state_dict(torch.load('best_model_weights_24_AAAI_BASELINE.pt'))
    next_val_4 = list()
    for num_iteration in range(num_samples_dropout):
        prediction_dropout = list()
        originial_testing_data = test_dataset
        for time in range(horizon):
            y_val_pred, h, c, embeddings = model(originial_testing_data)
            prediction_dropout.append(y_val_pred)
            to_be_added = torch.unsqueeze(y_val_pred, 0)
            next_data = torch.concatenate((originial_testing_data.x[1:, :, :], to_be_added))
            originial_testing_data.x = next_data
        prediction_dropout = torch.stack(prediction_dropout, 0)
        next_val_4.append(prediction_dropout)
    predictions_50_times = torch.stack(next_val_4, 0)
    predictions_50_times = torch.squeeze(predictions_50_times)
    all_50_predictions_for_all_timestamps.append(predictions_50_times)
all_50_predictions_for_all_timestamps = torch.stack(all_50_predictions_for_all_timestamps, 0)
torch.save(all_50_predictions_for_all_timestamps, '50_dropout_predictions_for_all_timestamps.pt')
