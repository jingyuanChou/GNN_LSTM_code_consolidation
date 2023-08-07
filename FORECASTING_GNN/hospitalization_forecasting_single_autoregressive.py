import matplotlib.pyplot as plt
import torch
import numpy as np
import argparse
import time
from util import *
from trainer import Trainer
from net import gtnet
import pandas as pd
import argparse
import math
import time
import networkx as nx
import torch
import torch.nn as nn
from net import gtnet
import numpy as np
import importlib
import tqdm
from util import *
from trainer import Optim

def str_to_bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in {'false', 'f', '0', 'no', 'n'}:
        return False
    elif value.lower() in {'true', 't', '1', 'yes', 'y'}:
        return True
    raise ValueError(f'{value} is not a valid boolean value')


parser = argparse.ArgumentParser()
parser.add_argument('--device',type=str,default='cpu',help='')
parser.add_argument('--data',type=str,default='data/METR-LA',help='data path')
parser.add_argument('--lr',type=float,default=0.01,help='learning rate')
parser.add_argument('--adj_data', type=str,default='data/sensor_graph/adj_mx.pkl',help='adj data path')
parser.add_argument('--gcn_true', type=str_to_bool, default=True, help='whether to add graph convolution layer')
parser.add_argument('--buildA_true', type=str_to_bool, default=True,help='whether to construct adaptive adjacency matrix')
parser.add_argument('--load_static_feature', type=str_to_bool, default=False,help='whether to load static feature')
parser.add_argument('--cl', type=str_to_bool, default=True,help='whether to do curriculum learning')
parser.add_argument('--horizon', type=int, default=1)
parser.add_argument('--prediction_window', type=int, default=4)
parser.add_argument('--step_size',type=int,default=100,help='step_size')
parser.add_argument('--save', type=str, default='model/model.pt',
                    help='path to save the final model')
parser.add_argument('--gcn_depth',type=int,default=1,help='graph convolution depth')
parser.add_argument('--num_nodes',type=int,default=51,help='number of nodes/variables')
parser.add_argument('--dropout',type=float,default=0.2,help='dropout rate')
parser.add_argument('--subgraph_size',type=int,default=10,help='k')
parser.add_argument('--node_dim',type=int,default=16,help='dim of nodes')
parser.add_argument('--dilation_exponential',type=int,default=1,help='dilation exponential')
parser.add_argument('--optim', type=str, default='adam')
parser.add_argument('--conv_channels',type=int,default=8,help='convolution channels')
parser.add_argument('--residual_channels',type=int,default=8,help='residual channels')
parser.add_argument('--skip_channels',type=int,default=16,help='skip channels')
parser.add_argument('--end_channels',type=int,default=32,help='end channels')
parser.add_argument('--in_dim',type=int,default=1,help='inputs dimension')
parser.add_argument('--seq_in_len',type=int,default=12,help='input sequence length')
parser.add_argument('--seq_out_len',type=int,default=1,help='output sequence length')
parser.add_argument('--layers',type=int,default=3,help='number of layers')
parser.add_argument('--batch_size',type=int,default=100,help='batch size')
parser.add_argument('--learning_rate',type=float,default=0.001,help='learning rate')
parser.add_argument('--weight_decay',type=float,default=0.0001,help='weight decay rate')
parser.add_argument('--clip',type=int,default=5,help='clip')
parser.add_argument('--step_size1',type=int,default=2500,help='step_size')
parser.add_argument('--step_size2',type=int,default=100,help='step_size')
parser.add_argument('--L1Loss', type=bool, default=True)
parser.add_argument('--epochs',type=int,default=200,help='')
parser.add_argument('--print_every',type=int,default=50,help='')
parser.add_argument('--seed',type=int,default=101,help='random seed')
parser.add_argument('--expid',type=int,default=1,help='experiment id')
parser.add_argument('--propalpha',type=float,default=0.05,help='prop alpha')
parser.add_argument('--tanhalpha',type=float,default=3,help='adj alpha')
parser.add_argument('--num_split',type=int,default=1,help='number of splits for graphs')
parser.add_argument('--normalize', type=int, default=0)
parser.add_argument('--runs',type=int,default=10,help='number of runs')
parser.add_argument('--autoregressive_retrain', type = bool, default=True, help='autoregressive training or not')


args = parser.parse_args()
torch.set_num_threads(3)

def evaluate(data, X, Y, model, criterion, batch_size):
    model.eval()
    total_loss = 0
    n_samples = 0
    predict = None
    test = None

    for X, Y in data.get_batches(X, Y, batch_size, False):
        X = torch.unsqueeze(X,dim=1)
        X = X.transpose(2,3)
        with torch.no_grad():
            output = model(X)
        output = torch.squeeze(output)
        if len(output.shape)==1:
            output = output.unsqueeze(dim=0)
        if predict is None:
            predict = output
            test = Y
        else:
            predict = torch.cat((predict, output))
            test = torch.cat((test, Y))

        scale = data.scale.expand(output.size(0), data.m)
        loss = criterion(output * scale, Y * scale)

        total_loss += loss.item()
        n_samples += (output.size(0) * data.m)

    return total_loss / n_samples

def predict_next_timestamp(data, X, model, future_timestamps, mc_num):
    model.train()
    mc_num_predictions = list()
    orginal_X = X
    for num_iteration in range(mc_num):
        X = orginal_X
        list_of_prediction = list()
        for i in range(future_timestamps):
            X = torch.unsqueeze(X, dim=1)
            if i == 0:
                X = X.transpose(2, 3)
            with torch.no_grad():
                output = model(X)
            X = torch.cat((X[:,:,:,1:], output), dim=3)
            output = torch.squeeze(output)
            X = torch.squeeze(X)
            if len(output.shape) == 1:
                output = output.unsqueeze(dim=0)
            scale = data.scale.expand(output.size(0), data.m)
            list_of_prediction.append((output * scale)[-1])
        next_4 = torch.stack(list_of_prediction, dim=1)
        mc_num_predictions.append(next_4)

    return mc_num_predictions




def train(data, X, Y, model, criterion, optim, batch_size):
    model.train()
    total_loss = 0
    n_samples = 0
    iter = 0
    for X, Y in data.get_batches(X, Y, batch_size, True):
        model.zero_grad()
        X = torch.unsqueeze(X,dim=1)
        X = X.transpose(2,3)
        if iter % args.step_size == 0:
            perm = np.random.permutation(range(args.num_nodes))
        num_sub = int(args.num_nodes / args.num_split)

        for j in range(args.num_split):
            if j != args.num_split - 1:
                id = perm[j * num_sub:(j + 1) * num_sub]
            else:
                id = perm[j * num_sub:]
            id = torch.tensor(id)
            tx = X[:, :, id, :]
            ty = Y[:, id]
            output = model(tx,id)
            output = torch.squeeze(output)
            scale = data.scale.expand(output.size(0), data.m)
            scale = scale[:,id]
            pred_Y = output * scale
            target_Y = ty * scale
            loss = criterion(output * scale, ty * scale)
            loss.backward()
            total_loss += loss.item()
            n_samples += (output.size(0) * data.m)
            grad_norm = optim.step()

    return total_loss / n_samples

def plot_for_each_state(Data, next_4, time):
    if time + 4 < len(Data):
        plot_data = Data[:(time + 4),:]
        available_data = Data[:time,:]
        for i in range(Data.shape[1]):
            pred_line = list(available_data[:,i])
            pred_line.extend(list(next_4[i].numpy()))
            target_line = list(plot_data[:, i])
            plt.plot(range(1, len(pred_line) + 1), pred_line)
            plt.plot(range(len(pred_line) - 4, len(pred_line) + 1), target_line[-5:])
            plt.show()
    return Data, next_4


def main(runid):
    # torch.manual_seed(args.seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    # np.random.seed(args.seed)
    #load data
    # args.buildA_true = False

    hopsitalization = pd.read_csv('hosp_weekly_filt_case_data.csv')
    hopsitalization = hopsitalization.iloc[:-1,:]
    hopsitalization = hopsitalization.reset_index()
    hopsitalization['state'] = hopsitalization['state'].astype(int)

    fips_list = hopsitalization['state'].values
    fips_2_index = hopsitalization.set_index('state')['index'].to_dict()
    index_2_fips = hopsitalization.set_index('index')['state'].to_dict()

    neighbouring_states = pd.read_csv('neighbouring_states.csv')
    fips_states = pd.read_csv('state_hhs_map.csv', header=None).iloc[:(-3),:]
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
    hopsitalization = hopsitalization.iloc[:,30:] # remove all 0 datapoints
    hopsitalization = hopsitalization.T.values



    list_of_next_4_since_current_time = list()
    adj = torch.from_numpy(adj.A)
    starting_time = 40
    for current_time in range(starting_time, hopsitalization.shape[0]):
        available_data = hopsitalization[:current_time,:]
        Data = DataLoaderS_hosp(available_data, 0.6, 0.2, args.device, args.horizon, args.seq_in_len,args.prediction_window, args.normalize)
        # if args.load_static_feature:
        #     static_feat = load_node_feature('data/sensor_graph/location.csv')
        # else:
        #     static_feat = None
        model = gtnet(args.gcn_true, args.buildA_true, args.gcn_depth, args.num_nodes,
                      args.device, adj, dropout=args.dropout, subgraph_size=args.subgraph_size,
                      node_dim=args.node_dim, dilation_exponential=args.dilation_exponential,
                      conv_channels=args.conv_channels, residual_channels=args.residual_channels,
                      skip_channels=args.skip_channels, end_channels=args.end_channels,
                      seq_length=args.seq_in_len, in_dim=args.in_dim, out_dim=args.seq_out_len,
                      layers=args.layers, propalpha=args.propalpha, tanhalpha=args.tanhalpha, layer_norm_affline=False)
        model = model.to(args.device)

        print(args)
        print('The recpetive field size is', model.receptive_field)
        nParams = sum([p.nelement() for p in model.parameters()])
        print('Number of model parameters is', nParams, flush=True)

        if args.L1Loss:
            criterion = nn.L1Loss().to(args.device)
        else:
            criterion = nn.MSELoss().to(args.device)
        evaluateL2 = nn.L1Loss().to(args.device)

        best_val = 10000000
        optim = Optim(
            model.parameters(), args.optim, args.lr, args.clip, lr_decay=args.weight_decay
        )

        # At any point you can hit Ctrl + C to break out of training early.

        print('begin training')
        train_losses = list()
        valid_losses = list()
        test_losses = list()
        for epoch in range(1, args.epochs + 1):
            train_loss = train(Data, Data.train[0], Data.train[1], model, criterion, optim, args.batch_size)
            val_loss = evaluate(Data, Data.valid[0], Data.valid[1], model, evaluateL2,
                                                   args.batch_size)
            # Save the model if the validation loss is the best we've seen so far.
            '''
            if val_loss < best_val:
                with open(args.save, 'wb') as f:
                    torch.save(model, f)
                best_val = val_loss
            '''
            test_loss = evaluate(Data, Data.test[0], Data.test[1], model, evaluateL2,
                                                         args.batch_size)
            train_losses.append(train_loss)
            valid_losses.append(val_loss)
            test_losses.append(test_loss)

            print(
                '| end of epoch {:3d} | train_loss {:5.4f} | val_loss {:5.4f} | test loss {:5.4f}'.format(
                    epoch, train_loss, val_loss, test_loss), flush=True)
        '''
        plt.plot(train_losses, label = 'train_loss')
        plt.plot(valid_losses, label = 'validation_loss')
        plt.plot(test_losses, label = 'testing_loss')
        plt.legend()
        plt.xlabel('epoch number')
        plt.ylabel('mae loss')
        plt.show()
        '''

        # Predict in autoregressive way
        next_4_pred = predict_next_timestamp(Data, Data.test[0], model, args.prediction_window)
        # plot_for_each_state(hopsitalization, next_4_pred, current_time)
        list_of_next_4_since_current_time.append(next_4_pred)

    plot_all(starting_time, list_of_next_4_since_current_time, hopsitalization)

    return list_of_next_4_since_current_time

def plot_all(time,list_of_next_4_since_current_time, hospitalization):
    cur_time = time
    list_of_next_4_since_current_time
    time_range = range(time, hospitalization.shape[0], 4)
    for i in range(hospitalization.shape[1]):
        plt.plot(hospitalization[:,i])
        for index, time in enumerate(time_range):
            if (time - cur_time) >= len(list_of_next_4_since_current_time):
                break
            x_value = list_of_next_4_since_current_time[time - cur_time][i].numpy()
            x_value = np.insert(x_value, 0, hospitalization[time,i])
            x_range = range(time, time + 5)
            plt.plot(x_range, x_value, color = 'orange')
        plt.show()


if __name__ == "__main__":
    vmae = []
    vmape = []
    vrmse = []
    mae = []
    mape = []
    rmse = []
    for i in range(args.runs):
        vm1, vm2, vm3, m1, m2, m3 = main(i)
        vmae.append(vm1)
        vmape.append(vm2)
        vrmse.append(vm3)
        mae.append(m1)
        mape.append(m2)
        rmse.append(m3)

    mae = np.array(mae)
    mape = np.array(mape)
    rmse = np.array(rmse)

    amae = np.mean(mae,0)
    amape = np.mean(mape,0)
    armse = np.mean(rmse,0)

    smae = np.std(mae,0)
    smape = np.std(mape,0)
    srmse = np.std(rmse,0)

    print('\n\nResults for 10 runs\n\n')
    #valid data
    print('valid\tMAE\tRMSE\tMAPE')
    log = 'mean:\t{:.4f}\t{:.4f}\t{:.4f}'
    print(log.format(np.mean(vmae),np.mean(vrmse),np.mean(vmape)))
    log = 'std:\t{:.4f}\t{:.4f}\t{:.4f}'
    print(log.format(np.std(vmae),np.std(vrmse),np.std(vmape)))
    print('\n\n')
    #test data
    print('test|horizon\tMAE-mean\tRMSE-mean\tMAPE-mean\tMAE-std\tRMSE-std\tMAPE-std')
    for i in [2,5,11]:
        log = '{:d}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}'
        print(log.format(i+1, amae[i], armse[i], amape[i], smae[i], srmse[i], smape[i]))





