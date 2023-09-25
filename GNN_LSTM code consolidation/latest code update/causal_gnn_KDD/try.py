import torch
from torch_geometric.data import Data
from torch_geometric.nn import GraphConv
import torch.nn.functional as F

# create toy data
data_list = [Data(x=torch.randn(3, 1)) for i in range(100)]

# create edge index
edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)


# create model
class GraphLSTMModel(torch.nn.Module):
    def __init__(self, num_nodes, input_dim, hidden_dim, output_dim):
        super(GraphLSTMModel, self).__init__()
        self.graphconv1 = GraphConv(input_dim, hidden_dim)
        self.lstm = torch.nn.LSTM(hidden_dim, hidden_dim)
        self.fc = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, x, edge_index):
        x = F.relu(self.graphconv1(x, edge_index))
        x = x.unsqueeze(0)
        x, _ = self.lstm(x)
        x = x.squeeze(0)
        x = self.fc(x)
        return x

if __name__ == '__main__':

    # set up model and hyperparameters
    num_nodes = 3
    input_dim = 1
    hidden_dim = 32
    output_dim = 1
    lr = 0.01
    num_epochs = 10
    sliding_window_size = 12

    model = GraphLSTMModel(num_nodes, input_dim, hidden_dim, output_dim)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # train the model
    for epoch in range(num_epochs):
        total_loss = 0
        for i in range(84):
            # get input and output data for sliding window
            input_data = [data.x for data in data_list[i:i + sliding_window_size]]
            output_data = [data.x for data in data_list[i + sliding_window_size:i + sliding_window_size + 4]]
            input_data = torch.stack(input_data, dim=0)
            output_data = torch.stack(output_data, dim=0)

            # make predictions and calculate loss
            predictions = model(input_data, edge_index)
            criterion.weight = torch.tensor([1, 1, 2, 2], dtype=torch.float)  # assign different weight to the timestamps
            loss = criterion(predictions, output_data)

            # backpropagation and weight updates
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print("Epoch {}, Loss {}".format(epoch, total_loss))

    # evaluate the model
    input_data = [data.x for data in data_list[84:96]]
    output_data = [data.x for data in data_list[96:100]]
    input_data = torch.stack(input_data, dim=0)
    output_data = torch.stack(output_data, dim=0)

    with torch.no_grad():
        predictions = model(input_data, edge_index)
        mse = criterion(predictions, output_data)

    print("MSE: {}".format(mse.item()))
