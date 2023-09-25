import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv
import numpy as np
import matplotlib.pyplot as plt

class GraphLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(GraphLSTMCell, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.gc1 = GCNConv(input_dim, hidden_dim)
        self.gc2 = GCNConv(hidden_dim, hidden_dim)
        self.lstm_cell = nn.LSTMCell(hidden_dim, hidden_dim)

    def forward(self, x, edge_index, h, c):
        x = self.gc1(x, edge_index)
        x = torch.relu(x)
        x = self.gc2(x, edge_index)
        h, c = self.lstm_cell(x, (h, c))
        return h, c

class GraphLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_nodes):
        super(GraphLSTM, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_nodes = num_nodes
        self.graph_lstm_cell = GraphLSTMCell(input_dim, hidden_dim)
        self.linear = nn.Linear(hidden_dim * num_nodes, num_nodes)

    def forward(self, x, edge_index, hidden_state, cell_state):
        h_seq = []
        for i in range(x.size(0)):
            hidden_state, cell_state = self.graph_lstm_cell(x[i], edge_index, hidden_state, cell_state)
            h_seq.append(hidden_state)
        h_seq = torch.stack(h_seq, dim=0)
        h_seq = h_seq.view(-1, self.hidden_dim * self.num_nodes)
        out = self.linear(h_seq)
        out = out.view(-1, self.num_nodes, 4)
        return out, hidden_state, cell_state

if __name__ == '__main__':
    # Define hyperparameters
    input_dim = 1
    hidden_dim = 16
    num_nodes = 3
    seq_len = 12
    pred_len = 4
    learning_rate = 0.001
    num_epochs = 10
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)


    # Generate toy data
    data = np.random.randn(100, 3, 1)

    # Split data into train and test sets
    train_data = data[:80, :, :]
    test_data = data[80:, :, :]

    # Create the model and optimizer
    model = GraphLSTM(input_dim, hidden_dim, num_nodes)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Train the model on a sliding window of data
    for i in range(seq_len, 100):
        # Extract the current window of input data
        x = data[i - seq_len:i, :, :]
        y = data[i:i + pred_len, :, :]
        x = torch.tensor(x, dtype=torch.float32).view(seq_len, num_nodes, input_dim)
        y = torch.tensor(y, dtype=torch.float32).view(pred_len, num_nodes, input_dim)

        # Initialize the hidden and cell states
        hidden_state = torch.zeros(num_nodes, hidden_dim)
        cell_state = torch.zeros(num_nodes, hidden_dim)

        # Run the model on the current window of data
        out, hidden_state, cell_state = model(x, edge_index, hidden_state, cell_state)

        # Compute the loss and perform backpropagation
        loss = torch.nn.functional.mse_loss(out, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Print the loss every 10 epochs
        # Define loss function and optimizer
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        # Train the model
        for epoch in range(num_epochs):
            for i in range(len(train_data) - seq_len - pred_len):
                # Extract input and target sequences
                inputs = train_data[i:i+seq_len, :, :]
                targets = train_data[i+seq_len:i+seq_len+pred_len, :, :]

                # Convert inputs and targets to PyTorch tensors
                inputs = torch.tensor(inputs, dtype=torch.float32)
                targets = torch.tensor(targets, dtype=torch.float32)

                # Forward pass
                outputs = model(inputs)

                # Compute loss and perform backward pass
                loss = criterion(outputs, targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Print loss every 100 steps
                if (i+1) % 100 == 0:
                    print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                          .format(epoch+1, num_epochs, i+1, len(train_data)-seq_len-pred_len, loss.item()))

                # Test the model
                model.eval()
                with torch.no_grad():
                    inputs = test_data[:seq_len, :, :]
                    targets = test_data[seq_len:seq_len+pred_len, :, :]
                    inputs = torch.tensor(inputs, dtype=torch.float32)
                    targets = torch.tensor(targets, dtype=torch.float32)
                    outputs = model(inputs)
                    test_loss = criterion(outputs, targets)
                    print('Test Loss: {:.4f}'.format(test_loss.item()))

                    # Plot predictions and actual values
                    plt.plot(targets.numpy(), label='Actual')
                    plt.plot(outputs.numpy(), label='Predicted')
                    plt.legend()
                    plt.show()

