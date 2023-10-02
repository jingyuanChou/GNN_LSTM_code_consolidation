import torch
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
import networkx as nx
from statsmodels.tsa.api import VAR
import pickle
def create_dataset(data, input_length, output_length):
    X, y = [], []
    for i in range(len(data) - input_length - output_length + 1):
        X.append(data[i:i + input_length, :])
        y.append(data[i + input_length:i + input_length + output_length, :])
    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

def generate_sequences(data, input_seq_len, pred_seq_len):
    input_sequences = []
    target_sequences = []
    for t in range(data.shape[0] - input_seq_len - pred_seq_len + 1):
        input_seq = data[t:t + input_seq_len,:, :]
        target_seq = data[t + input_seq_len:t + input_seq_len + pred_seq_len,:, 0]
        input_sequences.append(input_seq)
        target_sequences.append(target_seq)
    return torch.tensor(input_sequences, dtype=torch.float32), torch.tensor(target_sequences, dtype=torch.float32)
# Define the model
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h_n, _ = self.lstm(x)
        out = self.fc(h_n[:, -1, :])
        return out.view(x.size(0), 4, -1)
class GRUModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GRUModel, self).__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h_n, _ = self.gru(x)
        out = self.fc(h_n[:, -1, :])
        return out.view(x.size(0), 4, -1)
def Vector_Autoregression(data, past, horizon, index_2_fips, fips_2_state):
    # VAR
    # Assuming 'data' is a pandas DataFrame with your 51 variables and 119 timestamps
    data_x = data[:-4, :]
    data_y = data[-4:, :]
    model = VAR(data_x)
    fit_model = model.fit(maxlags=past)  # Choose an appropriate lag
    forecast = fit_model.forecast(data[-fit_model.k_ar:], steps=horizon)

    forecast = forecast.T
    gt = data_y.T

    list_mae_VAR = list()
    list_of_mape_VAR = list()

    VAR_MAE_dict = dict()

    for i in range(forecast.shape[0]):
        fips = index_2_fips[i]
        cur_state = fips_2_state[fips]
        VAR_MAE_dict[cur_state] = sum(abs(forecast[i] - gt[i]))/4
        list_mae_VAR.append(sum(abs(forecast[i] - gt[i])))
        sum_mape = 0
        for j, ele in enumerate(forecast[i]):
            temp_mape = abs((ele - gt[i][j]) / gt[i][j])
            sum_mape = sum_mape + temp_mape
        list_of_mape_VAR.append(sum_mape / 4)
    mean_mae_VAR = np.mean(list_mae_VAR)
    gt = np.sum(gt, axis=0)
    mean_mape_overall = np.sum(list_mae_VAR) / np.sum(gt)

    return VAR_MAE_dict
def LSTM(hopsitalization, input_length, output_length, hidden_dim, index_2_fips, fips_2_state):
    scaler = MinMaxScaler()
    hopsitalization = scaler.fit_transform(hopsitalization)

    input_dim = hopsitalization.shape[1]  # 49
    output_dim = output_length * input_dim  # 4 * 49

    length = hopsitalization.shape[0]
    train_idx = int(length * 0.8)
    val_idx = length - 4
    # Split data into training, validation, and test sets
    # [0-97] as training data, 98 timestamps, [98:118] as validation data, 21 timestamps , [102,122] total 21 timestamps
    train_data = hopsitalization[:train_idx, :]
    val_data = hopsitalization[train_idx:val_idx, :]
    test_data = hopsitalization[(train_idx + 4):, :]

    X_train, y_train = create_dataset(train_data, input_length, output_length)
    X_val, y_val = create_dataset(val_data, input_length, output_length)
    X_test, y_test = create_dataset(test_data, input_length, output_length)

    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    test_dataset = TensorDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=len(val_dataset))
    test_loader = DataLoader(test_dataset, batch_size=len(test_dataset))

    model = LSTMModel(input_dim, hidden_dim, output_dim)

    # Define loss and optimizer
    loss_function = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # Training loop
    epochs = 100
    training_loss = list()
    val_losses = list()
    for epoch in range(epochs):
        # Training
        model.train()
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            predictions = model(X_batch)
            loss = loss_function(predictions, y_batch)
            loss.backward()
            optimizer.step()
        # Validation
        model.eval()
        with torch.no_grad():
            val_loss = 0
            for X_batch, y_batch in val_loader:
                predictions = model(X_batch)
                val_loss += loss_function(predictions, y_batch).item()
            val_loss /= len(val_loader)
            print(f"Epoch {epoch + 1}/{epochs}, Training Loss: {loss} Validation Loss: {val_loss}")
        training_loss.append(loss.item())
        val_losses.append(val_loss)
    # Testing
    model.eval()
    with torch.no_grad():
        test_loss = 0
        for X_batch, y_batch in test_loader:
            predictions = model(X_batch)
            test_loss += loss_function(predictions, y_batch).item()
        test_loss /= len(test_loader)
        print(f"Test Loss: {test_loss}")
    predictions_reshaped = predictions.reshape(-1, 49).detach().numpy()
    predictions_original_scale = scaler.inverse_transform(predictions_reshaped)
    predictions_original_scale = predictions_original_scale.reshape(-1, 4, 49)
    last_output = predictions_original_scale[-1]

    targeted_label = y_batch[-1]

    loss_of_each_state = abs(targeted_label - last_output)

    LSTM_result = dict()

    for i in range(loss_of_each_state.shape[1]):
        cur_fips = index_2_fips[i]
        cur_state = fips_2_state[cur_fips]
        LSTM_result[cur_state] = torch.sum(loss_of_each_state[:, i]) / 4
    return LSTM_result
def GRU(hopsitalization, input_length, output_length, hidden_dim, index_2_fips, fips_2_state):
    scaler = MinMaxScaler()
    hopsitalization = scaler.fit_transform(hopsitalization)

    input_dim = hopsitalization.shape[1]  # 49
    output_dim = output_length * input_dim  # 4 * 49

    length = hopsitalization.shape[0]
    train_idx = int(length * 0.8)
    val_idx = length - 4
    # Split data into training, validation, and test sets
    # [0-97] as training data, 98 timestamps, [98:118] as validation data, 21 timestamps , [102,122] total 21 timestamps
    train_data = hopsitalization[:train_idx, :]
    val_data = hopsitalization[train_idx:val_idx, :]
    test_data = hopsitalization[(train_idx + 4):, :]

    X_train, y_train = create_dataset(train_data, input_length, output_length)
    X_val, y_val = create_dataset(val_data, input_length, output_length)
    X_test, y_test = create_dataset(test_data, input_length, output_length)

    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    test_dataset = TensorDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=len(val_dataset))
    test_loader = DataLoader(test_dataset, batch_size=len(test_dataset))

    model = GRUModel(input_dim, hidden_dim, output_dim)

    # Define loss and optimizer
    loss_function = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # Training loop
    epochs = 100
    training_loss = list()
    val_losses = list()
    for epoch in range(epochs):
        # Training
        model.train()
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            predictions = model(X_batch)
            loss = loss_function(predictions, y_batch)
            loss.backward()
            optimizer.step()
        # Validation
        model.eval()
        with torch.no_grad():
            val_loss = 0
            for X_batch, y_batch in val_loader:
                predictions = model(X_batch)
                val_loss += loss_function(predictions, y_batch).item()
            val_loss /= len(val_loader)
            print(f"Epoch {epoch + 1}/{epochs}, Training Loss: {loss} Validation Loss: {val_loss}")
        training_loss.append(loss.item())
        val_losses.append(val_loss)
    # Testing
    model.eval()
    with torch.no_grad():
        test_loss = 0
        for X_batch, y_batch in test_loader:
            predictions = model(X_batch)
            test_loss += loss_function(predictions, y_batch).item()
        test_loss /= len(test_loader)
        print(f"Test Loss: {test_loss}")
    predictions_reshaped = predictions.reshape(-1, 49).detach().numpy()
    predictions_original_scale = scaler.inverse_transform(predictions_reshaped)
    predictions_original_scale = predictions_original_scale.reshape(-1, 4, 49)
    last_output = predictions_original_scale[-1]

    targeted_label = y_batch[-1]

    loss_of_each_state = abs(targeted_label - last_output)

    GRU_result = dict()

    for i in range(loss_of_each_state.shape[1]):
        cur_fips = index_2_fips[i]
        cur_state = fips_2_state[cur_fips]
        GRU_result[cur_state] = torch.sum(loss_of_each_state[:, i]) / 4
    return GRU_result
def VAR_multi(data, past, horizon, index_2_fips, fips_2_state):
    forecast = Vector_Autoregression(data, past, horizon,index_2_fips, fips_2_state)
    return forecast

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

    hopsitalization = hopsitalization.iloc[:, 30:]  # remove all 0 datapoints
    hopsitalization = hopsitalization.T.values
    models = ['LSTM','GRU','VAR'] # GraphLSTM is in another repository, DCRNN as well
    input_length = 12
    output_length = 4
    hidden_dim = 8
    neighbouring_states_edges = neighbouring_states.iloc[0:109, :]
    for model in models:
        if model == 'LSTM':
            lstm_mae_result = LSTM(hopsitalization, input_length, output_length, hidden_dim, index_2_fips, fips_2_state)
            with open("lstm_result", "wb") as file:  # Pickling
                pickle.dump(lstm_mae_result, file)
        elif model == 'GRU':
            gru_mae_result = GRU(hopsitalization, input_length, output_length, hidden_dim, index_2_fips, fips_2_state)
            with open("gru_result", "wb") as file:  # Pickling
                pickle.dump(gru_mae_result, file)
        elif model == 'VAR':
            VAR_MAE_RESULT = VAR_multi(hopsitalization, input_length, output_length, index_2_fips, fips_2_state)
            with open("VAR_MAE_RESULT", "wb") as file:  # Pickling
                pickle.dump(VAR_MAE_RESULT, file)