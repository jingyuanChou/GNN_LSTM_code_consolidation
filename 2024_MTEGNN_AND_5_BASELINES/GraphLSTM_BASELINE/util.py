import torch
import numpy as np
import matplotlib.pyplot as plt
from torch_geometric.data import Data
from torch_geometric_temporal.nn import *

def mape_loss(output, label):
    return torch.mean(torch.abs(torch.div((output - label), label)))

def mse_loss(output, label, mean=None):
    return torch.mean(torch.square(output - label))

def msse_loss(output, label, mean=None):
    return torch.mean(torch.div(torch.square(output - label), label + 1))

def rmse_loss(output, label):
    return torch.sqrt(torch.mean(torch.square(output - label)))

def mae_loss(output, label):
    return torch.mean(torch.abs(output - label))

def mase_loss(output, label, mean=None):
    mean = mean.reshape(output.shape)
    label_mean = torch.mean(label)
    if not mean is None:
        return torch.mean(torch.abs(output - label) / mean)
    elif label_mean == 0:
        return torch.mean(torch.abs(output - label))
    else:
        return torch.mean(torch.abs(output - label)) / label_mean

def mase1_loss(output, label, mean=None):
    # Extreme 1: all countries equal
    # L_i = (x_i - y_i)^2 / y_i
    # L = (L_1 + L_2 + … + L_N) / N
    label = label[:, 0]
    output = output.reshape(output.shape[0]) # prediction of each state cases
    label_mean = torch.mean(label) # ground truth of each state abs(prediction - label) / ave(label)
    if not mean is None:
        return torch.mean(torch.abs(output - label) / mean)
    if label_mean == 0:
        return torch.mean(torch.abs(output - label))
    else:
        return torch.mean(torch.abs(output - label)) / label_mean

def MAE_loss(output, label):
    # abs(prediction - label) / number of elements
    # mean absolute error
    forecast_weeks = output.shape[2]
    num_states = output.shape[1]
    total_loss = 0

    for i in range(num_states):
        for j in range(forecast_weeks):
            total_loss = total_loss + torch.abs(output[0,i,j] - label[i,j])
    num_elements = forecast_weeks * num_states
    MAE = total_loss/num_elements
    return MAE




def mase2_loss(output, label, mean=None):
    # Extreme 2: all people equal
    # X = (x_1 + x_2 + … + x_N)
    # Y = (y_1 + y_2 + … + y_N)
    # L = (X - Y)^2 / Y
    label = label[:, 0]
    X = torch.sum(output)
    Y = torch.sum(label)
    if Y == 0 and not mean is None:
        return torch.abs(X - Y) / torch.sum(mean)
    elif Y == 0:
        return torch.abs(X - Y)
    else:
        return torch.abs(X - Y) / Y

def anti_lag_loss(output, label, lagged_label, mean=None, loss_func=mase2_loss, penalty_factor=0.1):
    output = output.reshape(output.shape[0])
    lagged_label = lagged_label.reshape(lagged_label.shape[0])

    # Or instead of penalty factor (or with it) should I be using the same loss function and taking the inverse square of that to ensure good scaling?
    penalty = torch.mean(torch.div(1, torch.square(output - lagged_label)))

    return loss_func(output, label, mean=mean) + penalty * penalty_factor

def lag_factor(output, lagged_label):
    return torch.div(torch.abs(output - lagged_label), lagged_label)

def mase3_loss(output, label, populations, mean=None, k=500000):
    # Middle point: consider a population threshold k
    # x_k = sum(x_i) such that country i has less than k population
    # y_k = sum(y_i) such that country i has less than k population
    # L_i = (x_i - y_i)^2 / y_i   for countries i with more than k population
    # L_k = (x_k - y_k)^2 / y_k
    # L = L_k + sum(L_i)
    label = label[:, 0]

    if mean is None:
        mean = torch.mean(label)
    if sum(mean) == 0:
        mean = 1

    large_outputs = []
    large_labels = []
    large_means = []

    small_outputs = []
    small_labels = []
    small_means = []
    for i in range(len(populations)):
        if populations[i] < k:
            small_outputs.append(output[i])
            small_labels.append(label[i])
            small_means.append(mean[i])
        else:
            large_outputs.append(output[i])
            large_labels.append(label[i])
            large_means.append(mean[i])

    x_k = sum(small_outputs)
    y_k = sum(small_labels)
    L_i = torch.abs(torch.FloatTensor(large_outputs) - torch.FloatTensor(large_labels)) / torch.FloatTensor(large_means)
    L_k = abs(x_k - y_k) / sum(small_means)
    return L_k + torch.sum(L_i)

def inv_reg_mase_loss(output, label):
    return mase_loss(output, label) + torch.mean(torch.div(1, output))

def train_gnn(model, loader, optimizer, loss_func, device):
    model.train()
    loss_all = 0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data)
        label = data.y.to(device)
        output = torch.reshape(output, label.shape)
        loss = loss_func(output, label)
        loss.backward()
        loss_all += data.num_graphs * loss.item()
        optimizer.step()
    return loss_all

def evaluate_gnn(model, loader, device):
    model.eval()
    predictions, labels = [], []
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            pred = model(data).detach().cpu().numpy()
            label = data.y.detach().cpu().numpy()
            pred = pred.reshape(label.shape)
            predictions.append(pred)
            labels.append(label)
    p = np.vstack(predictions)
    l = np.vstack(labels)
    return np.mean(np.abs(p - l)) / np.mean(l) #np.mean(abs((labels - predictions) / labels))  #reporting loss function, different from training


def evaluate_gnn_recurrent(model, dataset, lookback_pattern, loss_func):
    predictions, labels, losses = [], [], []

    def forward(snapshot, h, c, detach=False):
        if type(model) is GConvLSTM or type(model) is GConvGRU:
            h, c = model(snapshot.x, snapshot.edge_index, snapshot.edge_attr[:, 0], h, c)
            if detach:
                h = h.detach()
                c = c.detach()
            return h, h, c
        else:
            return model(snapshot, h, c)

    model.eval()
    with torch.no_grad():
        cost = 0
        for time, snapshot in enumerate(dataset):
            h, c = None, None
            for sub_time in range(len(lookback_pattern)):
                sub_snapshot = Data(x=snapshot.x[:, sub_time:sub_time + 1], edge_index=snapshot.edge_index,
                                    edge_attr=snapshot.edge_attr)
                y_hat, h, c = forward(sub_snapshot, h, c, detach=True)
            predictions.append(y_hat)
            labels.append(snapshot.y)
            cost += loss_func(y_hat, snapshot.y)
        cost /= time + 1
        cost = cost.item()
        losses.append(cost)
    return predictions, labels, losses

def show_predictions(predictions, labels):
    # Plot predictions and labels over time

    x = np.arange(0, len(predictions['train']))
    plt.title('COVID in US at State-wise level training set')
    plt.xlabel("Time (weeks)")
    plt.ylabel("New Cases")
    plt.plot(x, [torch.mean(p) for p in predictions['train']], label="Predictions", color = 'red')
    plt.plot(x, [torch.mean(l) for l in labels['train']], label="Labels", color = 'black')
    # plt.plot(x, [1000*mase_loss(predictions[i], labels[i]) for i in range(len(predictions))], label="Loss")
    plt.legend()
    plt.show()

    '''
    x = np.arange(0, len(predictions['val']))
    plt.title('COVID in US at State-wise level validation set')
    plt.xlabel("Time (weeks)")
    plt.ylabel("New Cases")
    plt.plot(x, [torch.mean(p) for p in predictions['val']], label="Predictions", color = 'red')
    plt.plot(x, [torch.mean(l) for l in labels['val']], label="Labels", color = 'black')
    # plt.plot(x, [1000*mase_loss(predictions[i], labels[i]) for i in range(len(predictions))], label="Loss")
    plt.legend()
    plt.show()
    '''

def show_loss_by_states(predictions, labels, nations, plot=True):
    losses = {}
    if plot:
        # Plot loss by country over time
        x = np.arange(0, len(predictions))
    fig, axs = plt.subplots(2,2)
    NY_index = nations.index('NY')
    CA_index = nations.index('CA')
    TX_index = nations.index('TX')
    VA_index = nations.index('VA')

    # NY
    if labels[0].shape[1] == 1:
        pred = [predictions[time][NY_index] for time in range(len(predictions))]
        label = [labels[time][NY_index] for time in range(len(predictions))]
        axs[0,0].plot(x, pred, label='{} predictions'.format(str(nations[NY_index])), color='blue')
        axs[0,0].plot(x, label, label='{} labels'.format(str(nations[NY_index])), color='red')
        axs[0,0].set_title('NEW YORK')
        axs[0,0].legend()
        axs[0, 0].set(ylabel='Number of Cases')


        # CA
        pred = [predictions[time][CA_index] for time in range(len(predictions))]
        label = [labels[time][CA_index] for time in range(len(predictions))]
        axs[0, 1].plot(x, pred, label='{} predictions'.format(str(nations[CA_index])), color='blue')
        axs[0, 1].plot(x, label, label='{} labels'.format(str(nations[CA_index])), color='red')
        axs[0,1].set_title('CALIFORNIA')
        axs[0, 1].legend()
        axs[0, 1].set(ylabel='Number of Cases')

        # TX
        pred = [predictions[time][TX_index] for time in range(len(predictions))]
        label = [labels[time][TX_index] for time in range(len(predictions))]
        axs[1, 0].plot(x, pred, label='{} predictions'.format(str(nations[TX_index])), color='blue')
        axs[1, 0].plot(x, label, label='{} labels'.format(str(nations[TX_index])), color='red')
        axs[1, 0].set_title('TEXAS')
        axs[1, 0].legend()
        axs[1, 0].set(xlabel='Weeks', ylabel='Number of Cases')


        # VA
        pred = [predictions[time][VA_index] for time in range(len(predictions))]
        label = [labels[time][VA_index] for time in range(len(predictions))]
        axs[1, 1].plot(x, pred, label='{} predictions'.format(str(nations[VA_index])), color='blue')
        axs[1, 1].plot(x, label, label='{} labels'.format(str(nations[VA_index])), color='red')
        axs[1, 1].set_title('VIRGINIA')
        axs[1, 1].legend()
        axs[1, 1].set(xlabel='Weeks', ylabel='Number of Cases')
    else:
        expected_labels = list()
        for ele in labels:
            candidate = ele[:,0]
            expected_labels.append(candidate)

        pred_candidate = list()
        for time in range(len(predictions)):
            temp_pred = predictions[time]
            pred_candidate.append(temp_pred[NY_index])

        label = [expected_labels[time][NY_index] for time in range(len(predictions))]
        axs[0, 0].plot(x, label, label='{} labels'.format(str(nations[NY_index])), color='red')
        axs[0, 0].set_title('NEW YORK')
        axs[0, 0].set(ylabel='Number of Cases')
        start = 0
        while start < len(predictions):
            xaxis = x[start: start+4]
            if len(xaxis) != len(pred_candidate[start]):
                break
            if start == 0:
                axs[0,0].plot(xaxis, pred_candidate[start], label='{} predictions'.format(str(nations[NY_index])), color='blue', linestyle = 'dashdot')
            else:
                axs[0,0].plot(xaxis, pred_candidate[start], color='blue', linestyle = 'dashdot')
            start = start + 3
        axs[0, 0].legend()

        # CA
        '''
        pred = [predictions[time][CA_index] for time in range(len(predictions))]
        label = [expected_labels[time][CA_index] for time in range(len(predictions))]
        axs[0, 1].plot(x, pred, label='{} predictions'.format(str(nations[CA_index])), color='blue')
        axs[0, 1].plot(x, label, label='{} labels'.format(str(nations[CA_index])), color='red')
        axs[0, 1].set_title('CALIFORNIA')
        axs[0, 1].legend()
        axs[0, 1].set(ylabel='Number of Cases')

        # TX
        pred = [predictions[time][TX_index] for time in range(len(predictions))]
        label = [expected_labels[time][TX_index] for time in range(len(predictions))]
        axs[1, 0].plot(x, pred, label='{} predictions'.format(str(nations[TX_index])), color='blue')
        axs[1, 0].plot(x, label, label='{} labels'.format(str(nations[TX_index])), color='red')
        axs[1, 0].set_title('TEXAS')
        axs[1, 0].legend()
        axs[1, 0].set(xlabel='Weeks', ylabel='Number of Cases')

        # VA
        pred = [predictions[time][VA_index] for time in range(len(predictions))]
        label = [expected_labels[time][VA_index] for time in range(len(predictions))]
        axs[1, 1].plot(x, pred, label='{} predictions'.format(str(nations[VA_index])), color='blue')
        axs[1, 1].plot(x, label, label='{} labels'.format(str(nations[VA_index])), color='red')
        axs[1, 1].set_title('VIRGINIA')
        axs[1, 1].legend()
        axs[1, 1].set(xlabel='Weeks', ylabel='Number of Cases')
        '''
    '''
    for i in range(len(nations)):
        plt.title('Loss by States')
        plt.xlabel("Time (weeks)")
        plt.ylabel("Cases per week")

        selected_states = ['NY','VA','CA','TX']
        if nations[i] in selected_states:
            print('ok')
        # Compute MAE loss for each example
            pred = [predictions[time][i] for time in range(len(predictions))]
            label = [labels[time][i] for time in range(len(predictions))]
            loss = [float(mae_loss(predictions[time][i], labels[time][i])) for time in range(len(predictions))]
            losses[nations[i]] = loss
            if plot:
                plt.plot(x, pred, label='{} predictions'.format(str(nations[i])), color='blue')
                plt.plot(x, label, label ='{} labels'.format(str(nations[i])), color = 'red')
                plt.legend()
    '''
    if plot:
        plt.show()
    return losses

def show_predictions_labels_by_states(predictions, labels, nations, plot=True):
    losses = {}

    train_prediction = predictions['train']
    val_prediction = predictions['val']
    # test_prediction = predictions['test']

    train_label = labels['train']
    val_label = labels['val']
    # test_label = labels['test']

    if plot:
        # Plot loss by country over time
        x = np.arange(0, len(train_prediction) + len(val_prediction))
    fig, axs = plt.subplots(2, 2)
    NY_index = nations.index('NY')
    CA_index = nations.index('CA')
    TX_index = nations.index('TX')
    VA_index = nations.index('VA')

    # NY
    pred_train = [train_prediction[time][NY_index] for time in range(len(train_prediction))]
    label_train = [train_label[time][NY_index] for time in range(len(train_label))]

    pred_val = [val_prediction[time][NY_index] for time in range(len(val_prediction))]
    label_val = [val_label[time][NY_index] for time in range(len(val_label))]

    #pred_test = [test_prediction[time][NY_index] for time in range(len(test_prediction))]
    #label_test = [test_label[time][NY_index] for time in range(len(test_label))]

    pred_train.extend(pred_val)
    #pred_train.extend(pred_test)

    label_train.extend(label_val)
    #label_train.extend(label_test)

    expected_labels = list()
    for ele in label_train:
        candidate = ele[0]
        expected_labels.append(candidate)

    axs[0, 0].plot(x, expected_labels, label='{} labels'.format(str(nations[NY_index])), color='black')
    axs[0, 0].set_title('NEW YORK')
    axs[0, 0].set(ylabel='Number of Cases')
    start = 0
    while start < len(pred_train):
        xaxis = x[start: start + 4]
        if len(xaxis) != len(pred_train[start]):
            # for single-week prediction
            axs[0, 0].plot(x, pred_train, label='{} predictions'.format(str(nations[NY_index])),
                           color='red', linestyle='dashdot')
            break
        if start == 0:
            axs[0, 0].plot(xaxis, pred_train[start], label='{} predictions'.format(str(nations[NY_index])),
                           color='red', linestyle='dashdot')
        else:
            axs[0, 0].plot(xaxis, pred_train[start], color='red', linestyle='dashdot')
        start = start + 3
    axs[0, 0].legend()

    # CA

    pred_train = [train_prediction[time][CA_index] for time in range(len(train_prediction))]
    label_train = [train_label[time][CA_index] for time in range(len(train_label))]

    pred_val = [val_prediction[time][CA_index] for time in range(len(val_prediction))]
    label_val = [val_label[time][CA_index] for time in range(len(val_label))]

    #pred_test = [test_prediction[time][CA_index] for time in range(len(test_prediction))]
    #label_test = [test_label[time][CA_index] for time in range(len(test_label))]

    pred_train.extend(pred_val)
    #pred_train.extend(pred_test)

    label_train.extend(label_val)
    #label_train.extend(label_test)

    expected_labels = list()
    for ele in label_train:
        candidate = ele[0]
        expected_labels.append(candidate)

    axs[0, 1].plot(x, expected_labels, label='{} labels'.format(str(nations[CA_index])), color='black')
    axs[0, 1].set_title('CALIFORNIA')
    axs[0, 1].set(ylabel='Number of Cases')
    start = 0
    while start < len(pred_train):
        xaxis = x[start: start + 4]
        if len(xaxis) != len(pred_train[start]):
            axs[0, 1].plot(x, pred_train, label='{} predictions'.format(str(nations[CA_index])),
                           color='red', linestyle='dashdot')
            break
        if start == 0:
            axs[0, 1].plot(xaxis, pred_train[start], label='{} predictions'.format(str(nations[CA_index])),
                           color='red', linestyle='dashdot')
        else:
            axs[0, 1].plot(xaxis, pred_train[start], color='red', linestyle='dashdot')
        start = start + 3
    axs[0, 1].legend()

    # TX
    pred_train = [train_prediction[time][TX_index] for time in range(len(train_prediction))]
    label_train = [train_label[time][TX_index] for time in range(len(train_label))]

    pred_val = [val_prediction[time][TX_index] for time in range(len(val_prediction))]
    label_val = [val_label[time][TX_index] for time in range(len(val_label))]

    #pred_test = [test_prediction[time][TX_index] for time in range(len(test_prediction))]
    #label_test = [test_label[time][TX_index] for time in range(len(test_label))]

    pred_train.extend(pred_val)
    #pred_train.extend(pred_test)

    label_train.extend(label_val)
    #label_train.extend(label_test)

    expected_labels = list()
    for ele in label_train:
        candidate = ele[0]
        expected_labels.append(candidate)

    axs[1, 0].plot(x, expected_labels, label='{} labels'.format(str(nations[TX_index])), color='black')
    axs[1, 0].set_title('TEXAS')
    axs[1, 0].set(xlabel='Weeks', ylabel='Number of Cases')
    start = 0
    while start < len(pred_train):
        xaxis = x[start: start + 4]
        if len(xaxis) != len(pred_train[start]):
            axs[1, 0].plot(x, pred_train, label='{} predictions'.format(str(nations[TX_index])),
                           color='red', linestyle='dashdot')
            break
        if start == 0:
            axs[1, 0].plot(xaxis, pred_train[start], label='{} predictions'.format(str(nations[TX_index])),
                           color='blue', linestyle='dashdot')
        else:
            axs[1, 0].plot(xaxis, pred_train[start], color='blue', linestyle='dashdot')
        start = start + 3
    axs[1, 0].legend()

    # VA
    pred_train = [train_prediction[time][VA_index] for time in range(len(train_prediction))]
    label_train = [train_label[time][VA_index] for time in range(len(train_label))]

    pred_val = [val_prediction[time][VA_index] for time in range(len(val_prediction))]
    label_val = [val_label[time][VA_index] for time in range(len(val_label))]

    #pred_test = [test_prediction[time][VA_index] for time in range(len(test_prediction))]
    #label_test = [test_label[time][VA_index] for time in range(len(test_label))]

    pred_train.extend(pred_val)
    #pred_train.extend(pred_test)

    label_train.extend(label_val)
    #label_train.extend(label_test)

    expected_labels = list()
    for ele in label_train:
        candidate = ele[0]
        expected_labels.append(candidate)

    axs[1, 1].plot(x, expected_labels, label='{} labels'.format(str(nations[VA_index])), color='black')
    axs[1, 1].set_title('TEXAS')
    axs[1, 1].set(xlabel='Weeks', ylabel='Number of Cases')
    start = 0
    while start < len(pred_train):
        xaxis = x[start: start + 4]
        if len(xaxis) != len(pred_train[start]):
            axs[1, 1].plot(x, pred_train, label='{} predictions'.format(str(nations[VA_index])),
                           color='red', linestyle='dashdot')
            break
        if start == 0:
            axs[1, 1].plot(xaxis, pred_train[start], label='{} predictions'.format(str(nations[VA_index])),
                           color='red', linestyle='dashdot')
        else:
            axs[1, 1].plot(xaxis, pred_train[start], color='red', linestyle='dashdot')
        start = start + 3
    axs[1, 1].legend()

    plt.show()

    if plot:
        plt.show()
    return losses

def show_labels_by_country(labels, nations):
    # Plot labels by country over time
    x = np.arange(0, len(labels))
    plt.title('New Cases by Country')
    plt.xlabel("Time (days)")
    plt.ylabel("New COVID Cases")
    for i in range(5):
        label = [torch.mean(l[i]) for l in labels]
        plt.plot(x, label, label=nations[i])
        print(nations[i] + ": " + str(int(sum(label)/len(label))))
    plt.show()