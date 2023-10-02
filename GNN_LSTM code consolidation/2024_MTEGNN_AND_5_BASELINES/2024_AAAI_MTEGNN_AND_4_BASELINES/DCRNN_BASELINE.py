import networkx as nx
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from torch_geometric.utils import degree, to_dense_adj
from torch_geometric.utils.convert import from_networkx
import pickle
def generate_sequences(data, input_seq_len, pred_seq_len):
    input_sequences = []
    target_sequences = []
    for t in range(data.shape[0] - input_seq_len - pred_seq_len + 1):
        input_seq = data[t:t + input_seq_len,:, :]
        target_seq = data[t + input_seq_len:t + input_seq_len + pred_seq_len,:, 0]
        input_sequences.append(input_seq)
        target_sequences.append(target_seq)
    return torch.tensor(input_sequences, dtype=torch.float32), torch.tensor(target_sequences, dtype=torch.float32)


# DCRNN Components
class DCRNNCell(nn.Module):
    def __init__(self, node_features, hidden_channels, K):
        super(DCRNNCell, self).__init__()
        self.dc_initial = nn.Linear((K+1) * node_features, hidden_channels)
        self.dc_hidden = nn.Linear((K+1) * hidden_channels, hidden_channels)

    def forward(self, x, edge_index, h=None):
        L = normalized_laplacian(edge_index, x.size(0))
        x_list = [x]
        for k in range(K):
            x = torch.matmul(L, x)
            x_list.append(x)
        x_concat = torch.cat(x_list, dim=-1)
        if x.size(-1) == 1:
            h_next = torch.relu(self.dc_initial(x_concat))
        else:
            h_next = torch.relu(self.dc_hidden(x_concat))
        return h_next


# Traffic Predictor
class TrafficPredictor(nn.Module):
    def __init__(self, node_features, hidden_channels, K):
        super(TrafficPredictor, self).__init__()
        self.dcrnn = DCRNNCell(node_features, hidden_channels, K)
        self.linear = nn.Linear(hidden_channels, 1)

    def forward(self, x_sequence, edge_index):
        # x_sequence: [batch_size, seq_len, num_nodes, feature_dim]
        batch_size, seq_length, num_nodes, _ = x_sequence.shape
        outputs = []

        for i in range(batch_size):
            h = None
            for t in range(seq_length):
                h = self.dcrnn(x_sequence[i, t, :, :], edge_index)

            predictions = []
            for _ in range(4):  # predict next 4 timestamps
                h = self.dcrnn(h, edge_index)
                pred = self.linear(h)
                predictions.append(pred)

            outputs.append(torch.stack(predictions, dim=0))

        return torch.stack(outputs, dim=0)

# Normalized Laplacian Function

def normalized_laplacian(edge_index, num_nodes):
    A = to_dense_adj(edge_index).squeeze(0)
    deg = degree(edge_index[0]).float()
    D_inv_sqrt = torch.diag(1.0 / torch.sqrt(deg))
    L = torch.eye(num_nodes) - torch.mm(torch.mm(D_inv_sqrt, A), D_inv_sqrt)
    return L


# Hyperparameters and Training

if __name__ == '__main__':

    hopsitalization = pd.read_csv('hosp_weekly_filt_case_data.csv')
    hopsitalization = hopsitalization.reset_index()
    hopsitalization['state'] = hopsitalization['state'].astype(int)

    fips_list = hopsitalization['state'].values
    fips_2_index = hopsitalization.set_index('state')['index'].to_dict()
    index_2_fips = hopsitalization.set_index('index')['state'].to_dict()

    neighbouring_states = pd.read_csv('neighbouring_states.csv')
    states_code = pd.read_csv('state_hhs_map.csv')

    neighbouring_states_copy = neighbouring_states[['NeighborStateCode', 'StateCode']]
    neighbouring_states_copy.columns = ['StateCode', 'NeighborStateCode']

    neighbouring_states = neighbouring_states.append(neighbouring_states_copy)
    neighbouring_states.drop_duplicates()

    fips_states = pd.read_csv('state_hhs_map.csv', header=None)
    fips_2_state = fips_states.set_index(0)[2].to_dict()
    state_2_fips = fips_states.set_index(2)[0].to_dict()
    hopsitalization['state'] = hopsitalization['state'].map(fips_2_state)

    index_2_state = hopsitalization.set_index('index')['state'].to_dict()
    state_2_index = hopsitalization.set_index('state')['index'].to_dict()

    neighbouring_states['StateCode'] = neighbouring_states['StateCode'].map(state_2_index)
    neighbouring_states['NeighborStateCode'] = neighbouring_states['NeighborStateCode'].map(state_2_index)

    G = nx.from_pandas_edgelist(neighbouring_states, 'StateCode', 'NeighborStateCode')
    Pyg_graph = from_networkx(G)
    edge_index = Pyg_graph.edge_index
    hopsitalization = hopsitalization.iloc[:, 30:]  # remove all 0 datapoints
    hopsitalization = hopsitalization.T.values

    input_length = 12
    output_length = 4
    hidden_dim = 8

    neighbouring_states_edges = neighbouring_states.iloc[0:109, :]

    scaler = MinMaxScaler()
    hopsitalization = scaler.fit_transform(hopsitalization)

    node_features = 1
    num_nodes = 49

    length = hopsitalization.shape[0]
    train_idx = int(length * 0.8)
    val_idx = length - 4
    # Split data into training, validation, and test sets
    # [0-97] as training data, 98 timestamps, [98:118] as validation data, 21 timestamps , [102,122] total 21 timestamps
    hopsitalization = np.expand_dims(hopsitalization, -1)
    train_data = hopsitalization[:train_idx, :]
    val_data = hopsitalization[train_idx:val_idx, :]
    test_data = hopsitalization[(train_idx + 4):, :]

    X_train, y_train = generate_sequences(train_data, input_length, output_length)
    X_val, y_val = generate_sequences(val_data, input_length, output_length)
    X_test, y_test = generate_sequences(test_data, input_length, output_length)

    learning_rate = 0.001
    hidden_dim = 8
    K = 2
    num_layers = 2
    epochs = 100

    model = TrafficPredictor(1, hidden_dim, K)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.L1Loss()

    min_val_loss = float('inf')
    best_model_path = 'best_model_DCRNN.pth'
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        predictions_train = model(X_train, edge_index)
        predictions_train = torch.squeeze(predictions_train)# [83, 4, 49, 1]
        loss_train = criterion(predictions_train, y_train)
        loss_train.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            predictions_val = model(X_val, edge_index)
            predictions_val = torch.squeeze(predictions_val)# [6, 4, 49, 1]
            loss_val = criterion(predictions_val, y_val)

        print(
            f"Epoch {epoch + 1}/{epochs} - Train Loss: {loss_train.item():.4f}, Validation Loss: {loss_val.item():.4f}")

        if loss_val < min_val_loss and loss_train > loss_val:
            min_val_loss = loss_val
            torch.save(model.state_dict(), best_model_path)
            best_epoch = epoch

    # Prediction stage
    if min_val_loss == float('inf'):
        best_DCRNN_model = model
    else:
        best_DCRNN_model = TrafficPredictor(1, hidden_dim, K)
        best_DCRNN_model.load_state_dict(torch.load(best_model_path))
    predictions_test = best_DCRNN_model(X_test, edge_index)
    y_test = torch.unsqueeze(y_test,3)  # [6, 4, 49, 1]
    loss_val = criterion(predictions_test, y_test)
    final_pred = y_test[-1]
    final_pred = torch.squeeze(final_pred)
    reverse_back_value = scaler.inverse_transform(final_pred)
    final_4_steps = final_pred.detach().cpu().numpy()
    # check loss
    state_MAE_dict = dict()

    for index in range(final_4_steps.shape[1]):
        cur_state = index_2_state[index]
        ave_loss_per_state = np.sum(abs(final_4_steps[:,index] - reverse_back_value[:,index])) / 4
        state_MAE_dict[cur_state] = ave_loss_per_state
    with open("DCRNN_result", "wb") as dcrnn_res:  # Pickling
        pickle.dump(state_MAE_dict, dcrnn_res)








