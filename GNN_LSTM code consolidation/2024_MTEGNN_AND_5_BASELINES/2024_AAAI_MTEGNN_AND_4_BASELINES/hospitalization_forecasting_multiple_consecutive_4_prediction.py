import matplotlib.pyplot as plt
import torch
import numpy as np
import argparse
import time
from util import *
from trainer import Trainer
import pandas as pd
import argparse
import math
import time
import networkx as nx
import torch
import torch.nn as nn
from net import gtnet
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
parser.add_argument('--horizon', type=int, default=4)
parser.add_argument('--prediction_window', type=int, default=4)
parser.add_argument('--step_size',type=int,default=100,help='step_size')
parser.add_argument('--save', type=str, default='model/model.pt',
                    help='path to save the final model')
parser.add_argument('--gcn_depth',type=int,default=1,help='graph convolution depth')
parser.add_argument('--num_nodes',type=int,default=49,help='number of nodes/variables')
parser.add_argument('--dropout',type=float,default=0.2,help='dropout rate')
parser.add_argument('--subgraph_size',type=int,default=20,help='k')
parser.add_argument('--node_dim',type=int,default=16,help='dim of nodes')
parser.add_argument('--init_embedding_dim',type=int,default=49,help='dim of embedding')
parser.add_argument('--dilation_exponential',type=int,default=1,help='dilation exponential')
parser.add_argument('--optim', type=str, default='adam')
parser.add_argument('--conv_channels',type=int,default=4,help='convolution channels')
parser.add_argument('--residual_channels',type=int,default=4,help='residual channels')
parser.add_argument('--skip_channels',type=int,default=8,help='skip channels')
parser.add_argument('--end_channels',type=int,default=16,help='end channels')
parser.add_argument('--in_dim',type=int,default=1,help='inputs dimension')
parser.add_argument('--seq_in_len',type=int,default=12,help='input sequence length')
parser.add_argument('--seq_out_len',type=int,default=4,help='output sequence length')
parser.add_argument('--layers',type=int,default=3,help='number of layers')
parser.add_argument('--batch_size',type=int,default=100,help='batch size')
parser.add_argument('--learning_rate',type=float,default=0.005,help='learning rate')
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
parser.add_argument('--tanhalpha',type=float,default=1,help='adj alpha')
parser.add_argument('--num_split',type=int,default=1,help='number of splits for graphs')
parser.add_argument('--normalize', type=int, default=0)
parser.add_argument('--runs',type=int,default=10,help='number of runs')
parser.add_argument('--autoregressive_retrain', type = bool, default=True, help='autoregressive training or not')
parser.add_argument('--multivariate_TE_enhanced', type = bool, default=True, help='Multivariate Transfer Entropy')
parser.add_argument('--Plot_4_smaller_regions', type = bool, default=True, help='Plot for smaller regions')

args = parser.parse_args()
torch.set_num_threads(3)

def evaluate(data, X, Y, model, criterion, batch_size):
    model.eval()
    total_loss = 0
    n_samples = 0
    predict = None
    test = None
    adj_test_ls = list()
    for X, Y in data.get_batches(X, Y, batch_size, False):
        X = torch.unsqueeze(X,dim=1)
        X = X.transpose(2,3)
        with torch.no_grad():
            output, adj_test = model(X)
        output = torch.squeeze(output)
        if len(output.shape)==1:
            output = output.unsqueeze(dim=0)
        if predict is None:
            predict = output
            test = Y
        else:
            predict = torch.cat((predict, output))
            test = torch.cat((test, Y))
        loss = criterion(output, Y)
        total_loss += loss.item()
        n_samples += (output.size(0) * output.size(1) * data.m)
        adj_test_ls.append(adj_test)
    return total_loss / n_samples, adj_test_ls

def predict_next_timestamp(X, model, mc_num):
    model.train() # enable dropout
    mc_num_predictions = list()
    orginal_X = X # number of testing samples * timestamps * number of regions
    adj_mc_num = list()
    for num_iteration in range(mc_num):
        X = orginal_X
        X = torch.unsqueeze(X, dim=1)
        X = X.transpose(2, 3)
        with torch.no_grad():
            output,adj = model(X)
        output = output.transpose(1,2)
        output = output[-1]
        output = torch.squeeze(output).numpy()
        mc_num_predictions.append(output)
        adj_mc_num.append(adj)
    return mc_num_predictions, adj_mc_num

def train(data, X, Y, model, criterion, optim, batch_size):
    model.train()
    total_loss = 0
    n_samples = 0
    iter = 0
    adj_list = list()
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
            ty = Y[:, :, id]
            output, adp = model(tx,id)
            output = torch.squeeze(output)
            loss = criterion(output, ty)
            loss.backward()
            total_loss += loss.item()
            n_samples += (output.size(0) * output.size(1) * data.m)
            grad_norm = optim.step()
        adj_list.append(adp)
    return total_loss / n_samples, adj_list

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


def main(exp_name,exp_num, best_20_runs_train_adj_list, best_epoch_20_runs):
    if exp_name == 'geo': # Geo-static
        args.buildA_true = False
        args.multivariate_TE_enhanced = False
    elif exp_name == 'puredata': # Data-driven, Original MTGNN
        args.buildA_true = True
        args.multivariate_TE_enhanced = False
    elif exp_name == 'MTE_based': # MTE-adaptive, update matrix accordingly
        args.buildA_true = True
        args.multivariate_TE_enhanced = True
    else: # MTE-static, no update
        args.buildA_true = False
        args.multivariate_TE_enhanced = True

    # args.Plot_4_smaller_regions = True

    hopsitalization = pd.read_csv('hosp_weekly_filt_case_data.csv')
    hopsitalization = hopsitalization.iloc[:-1,:]
    hopsitalization = hopsitalization.reset_index()
    hopsitalization['state'] = hopsitalization['state'].astype(int)
    fips_list = hopsitalization['state'].values
    fips_2_index = hopsitalization.set_index('state')['index'].to_dict()
    index_2_fips = hopsitalization.set_index('index')['state'].to_dict()
    neighbouring_states = pd.read_csv('neighbouring_states.csv')
    fips_states = pd.read_csv('state_hhs_map.csv', header=None).iloc[:(-3),:]
    fips_2_state = fips_states.set_index(0)[2].to_dict()
    state_2_fips = fips_states.set_index(2)[0].to_dict()
    state_len = len(fips_list)
    hopsitalization['state'] = hopsitalization['state'].map(fips_2_state)
    index_2_state = hopsitalization.set_index('index')['state'].to_dict()
    state_2_index = hopsitalization.set_index('state')['index'].to_dict()

    neighbouring_states['StateCode'] = neighbouring_states['StateCode'].map(state_2_index)
    neighbouring_states['NeighborStateCode'] = neighbouring_states['NeighborStateCode'].map(state_2_index)

    G = nx.from_pandas_edgelist(neighbouring_states, 'StateCode', 'NeighborStateCode')
    hopsitalization = hopsitalization.iloc[:,30:] # remove all 0 datapoints
    hopsitalization = hopsitalization.T.values

    if exp_name in ['MTE_based', 'MTE_static']:
        adj = load_TE_matrix(args.multivariate_TE_enhanced)
        # adj_df = pd.DataFrame(adj)
        adj = torch.from_numpy(adj)
    elif exp_name == 'geo':
        adj = nx.adjacency_matrix(G)
        adj = torch.from_numpy(adj.A)
    else:
        adj = None
    current_time = 119 # using 0-119 to predict 120-123

    available_data = hopsitalization[:current_time,:]
    final_20_run_results = dict()
    list_of_next_4_since_current_time = list()

    if args.Plot_4_smaller_regions:
        with open("Multistep results/geo_static_final_pred_subgraph_20_and_alpha_1_expnum_10", "rb") as fp:  # Unpickling
            geo_result = pickle.load(fp)

        with open('Multistep results/MTGNN_adaptive_final_pred_subgraph_20_and_alpha_1_expnum_10','rb') as mtgnn_pure:
            mtgnn_result = pickle.load(mtgnn_pure)

        with open('Multistep results/MTE_static_final_pred_subgraph_20_and_alpha_1_expnum_10','rb') as MTE_static:
            MTE_static = pickle.load(MTE_static)

        with open('Multistep results/MTE_adaptive_final_pred_subgraph_20_and_alpha_1_expnum_10','rb') as Te_result:
            Te_result = pickle.load(Te_result)

        for i in range(hopsitalization.shape[1]):
            x_range = range(hopsitalization.shape[0] - 10 + 1, hopsitalization.shape[0] + 1)
            plt.plot(x_range, hopsitalization[-10:, i], marker='>', markersize=2.0, label='ground truth', color = 'Black')
            prev_6_timestamps = hopsitalization[-10:-4, i]
            next_4_timestamps_geo = geo_result[i]
            next_4_timestamps_mtgnn = mtgnn_result[i]
            next_4_timestamps_TEresult = Te_result[i]
            next_4_timestamps_TE_static = MTE_static[i]

            x_value_geo = np.concatenate((prev_6_timestamps, next_4_timestamps_geo), axis=None)
            x_value_mtgnn = np.concatenate((prev_6_timestamps, next_4_timestamps_mtgnn), axis=None)
            x_value_TEresult = np.concatenate((prev_6_timestamps, next_4_timestamps_TEresult), axis=None)
            x_value_TE_static = np.concatenate((prev_6_timestamps, next_4_timestamps_TE_static), axis=None)

            plt.plot(x_range[5:], x_value_mtgnn[5:], linestyle='--', marker='>',
                     markersize=2.0, label='MTGNN', color = 'Blue')
            plt.plot(x_range[5:], x_value_geo[5:], linestyle='--', marker='o',
                     markersize=2.0, label='Geo-MTGNN',color = 'Green')
            plt.plot(x_range[5:], x_value_TEresult[5:], linestyle='--', marker='<',
                     markersize=2.0, label='MTE-MTGNN',color='red')
            plt.plot(x_range[5:], x_value_TE_static[5:], linestyle='--', marker='s',
                     markersize=2.0, label='MTE_static',color='m')

            plt.title('{} prediction'.format(index_2_state[i]))
            plt.xlabel('Time (Weeks)')
            plt.ylabel('# Hospitalizations')
            plt.legend()
            plt.show()

    for iter in range(exp_num):
        Data = DataLoaderM_hosp(available_data, 0.8, 0.2, args.device, args.horizon, args.seq_in_len,args.prediction_window, args.normalize)
        model = gtnet(args.gcn_true, args.buildA_true, args.gcn_depth, args.num_nodes,
                      args.device, adj, dropout=args.dropout, subgraph_size=args.subgraph_size,
                      node_dim=args.node_dim, init_emb_dim = args.init_embedding_dim, dilation_exponential=args.dilation_exponential,
                      conv_channels=args.conv_channels, residual_channels=args.residual_channels,
                      skip_channels=args.skip_channels, end_channels=args.end_channels,
                      seq_length=args.seq_in_len, in_dim=args.in_dim, out_dim=args.seq_out_len,
                      layers=args.layers, propalpha=args.propalpha, tanhalpha=args.tanhalpha, layer_norm_affline=False, MTE=args.multivariate_TE_enhanced)
        model = model.to(args.device)

        print('The recpetive field size is', model.receptive_field)
        nParams = sum([p.nelement() for p in model.parameters()])
        print('Number of model parameters is', nParams, flush=True)

        if args.L1Loss:
            criterion = nn.L1Loss().to(args.device)
        else:
            criterion = nn.MSELoss().to(args.device)

        optim = Optim(
            model.parameters(), args.optim, args.lr, args.clip, lr_decay=args.weight_decay
        )

        # At any point you can hit Ctrl + C to break out of training early.

        print('begin training')
        train_losses = list()
        valid_losses = list()

        min_val_loss = float('inf')
        best_model_path = 'best_model.pth'
        for epoch in range(1, args.epochs + 1):
            train_loss, adj_list = train(Data, Data.train[0], Data.train[1], model, criterion, optim, args.batch_size)
            val_loss, adj_list_val = evaluate(Data, Data.valid[0], Data.valid[1], model, criterion,
                                                   args.batch_size)
            # Save the model if the validation loss is the best we've seen so far.
            '''
            if val_loss < best_val:
                with open(args.save, 'wb') as f:
                    torch.save(model, f)
                best_val = val_loss
            '''
            train_losses.append(train_loss)
            valid_losses.append(val_loss)
            if val_loss < min_val_loss:
                min_val_loss = val_loss
                # Save this model
                torch.save(model.state_dict(), best_model_path)
                best_adj_list = adj_list
                best_epoch = epoch

            print(
                '| end of epoch {:3d} | train_loss {:5.4f} | val_loss {:5.4f}'.format(
                    epoch, train_loss, val_loss), flush=True)
        '''
        plt.plot(train_losses, label = 'train_loss')
        plt.plot(valid_losses, label = 'validation_loss')
        plt.legend()
        plt.xlabel('epoch number')
        plt.ylabel('mae loss per sample')
        plt.show()
        '''
        best_model = gtnet(args.gcn_true, args.buildA_true, args.gcn_depth, args.num_nodes,
                           args.device, adj, dropout=args.dropout, subgraph_size=args.subgraph_size,
                           node_dim=args.node_dim, dilation_exponential=args.dilation_exponential,
                           conv_channels=args.conv_channels, residual_channels=args.residual_channels,
                           skip_channels=args.skip_channels, end_channels=args.end_channels,
                           seq_length=args.seq_in_len, in_dim=args.in_dim, out_dim=args.seq_out_len,
                           layers=args.layers, propalpha=args.propalpha, tanhalpha=args.tanhalpha,
                           layer_norm_affline=False, MTE=args.multivariate_TE_enhanced)
        best_model.load_state_dict(torch.load(best_model_path))

        # Predict in autoregressive way
        next_4_pred, adj_test_list = predict_next_timestamp(Data.predict_data, best_model, 50)
        mae_list, mape_list = mae_mape_calculation(next_4_pred, hopsitalization)
        if iter == 0:
            final_20_run_results['mae'] = list()
            final_20_run_results['mape'] = list()
        final_20_run_results['mae'].append(mae_list)
        final_20_run_results['mape'].append(mape_list)
        list_of_next_4_since_current_time.append(next_4_pred)
    if exp_name =='MTE_based':
        best_epoch_20_runs.append(best_epoch)
        best_20_runs_train_adj_list= best_adj_list[0].detach().numpy()
    # plot_all(list_of_next_4_since_current_time, hopsitalization, index_2_state)

    list_of_next_4_since_current_time = [np.mean(list_of_next_4_since_current_time[i], axis=0) for i in
                                         range(len(list_of_next_4_since_current_time))]
    list_of_next_4_since_current_time = np.mean(list_of_next_4_since_current_time, axis=0)

    if not args.buildA_true:
        if exp_name == 'geo': # geo matrix based, without graph learning
            with open("geo_static_performance_result_subgraph_size_{}_and_alpha_{}_expnum_{}".format(args.subgraph_size, args.tanhalpha,exp_num), "wb") as geo:  # Pickling
                pickle.dump(final_20_run_results, geo)
            with open("geo_static_final_pred_subgraph_{}_and_alpha_{}_expnum_{}".format(args.subgraph_size,args.tanhalpha,exp_num), "wb") as geo_result:  # Pickling
                pickle.dump(list_of_next_4_since_current_time, geo_result)
            with open("geo_static_adj_matrix_subgraph_{}_and_alpha_{}_expnum_{}".format(args.subgraph_size,args.tanhalpha,exp_num), "wb") as geo_result_adj:  # Pickling
                pickle.dump(best_20_runs_train_adj_list, geo_result_adj)
        else: # MTE static based, skipped the graph update procedure, without graph learning
            with open("MTE_static_performance_result_subgraph_size_{}_and_{}_expnum_{}".format(args.subgraph_size, args.tanhalpha,exp_num), "wb") as MTE_static:  # Pickling
                pickle.dump(final_20_run_results, MTE_static)
            with open("MTE_static_final_pred_subgraph_{}_and_alpha_{}_expnum_{}".format(args.subgraph_size,args.tanhalpha,exp_num), "wb") as MTE_static_result:  # Pickling
                pickle.dump(list_of_next_4_since_current_time, MTE_static_result)
            with open("MTE_static_adj_matrix_subgraph_{}_and_alpha_{}_expnum_{}".format(args.subgraph_size,args.tanhalpha,exp_num), "wb") as MTE_static_adj:  # Pickling
                pickle.dump(best_20_runs_train_adj_list, MTE_static_adj)

    else:
        if args.multivariate_TE_enhanced: # MTE enhanced, with graph learning.
            with open("MTE_adaptive_performance_result_subgraph_size_{}_and_{}_expnum_{}".format(args.subgraph_size, args.tanhalpha, exp_num), "wb") as DD:  # Pickling
                pickle.dump(final_20_run_results, DD)
            with open("MTE_adaptive_final_pred_subgraph_{}_and_alpha_{}_expnum_{}".format(args.subgraph_size, args.tanhalpha, exp_num), "wb") as DD_result:  # Pickling
                pickle.dump(list_of_next_4_since_current_time, DD_result)
            with open("MTE_adaptive_adj_matrix_subgraph_{}_and_alpha_{}_expnum_{}".format(args.subgraph_size, args.tanhalpha, exp_num), "wb") as MTE_result_adj:  # Pickling
                pickle.dump(best_20_runs_train_adj_list, MTE_result_adj)
        else: # Pure MTGNN, with graph learning
            with open("MTGNN_adaptive_performance_result_subgraph_size_{}_and_{}_expnum_{}".format(args.subgraph_size, args.tanhalpha,exp_num), "wb") as PUREDATA:  # Pickling
                pickle.dump(final_20_run_results, PUREDATA)
            with open("MTGNN_adaptive_final_pred_subgraph_{}_and_alpha_{}_expnum_{}".format(args.subgraph_size, args.tanhalpha, exp_num), "wb") as pure_data_result:  # Pickling
                pickle.dump(list_of_next_4_since_current_time, pure_data_result)
            with open("MTGNN_adaptive_adj_matrix_subgraph_{}_and_alpha_{}_expnum_{}".format(args.subgraph_size, args.tanhalpha, exp_num), "wb") as pure_data_result_adj:  # Pickling
                pickle.dump(best_20_runs_train_adj_list, pure_data_result_adj)
    return


def mae_mape_calculation(next_4_pred, hospitalization):
    list_of_mae = list()
    list_of_mape = list()
    for i in range(hospitalization.shape[1]): # i-th region
        temp_i = list()
        for j in range(len(next_4_pred)):
            temp_i.append(next_4_pred[j][i])
        mean = np.mean(temp_i, axis=0)
        gt = hospitalization[-4:, i]
        pred = mean
        list_of_mae.append(sum(abs(pred - gt)))
        sum_mape = 0
        for j, ele in enumerate(pred):
            temp_mape = abs((ele - gt[j])/gt[j])
            sum_mape = sum_mape+temp_mape
        list_of_mape.append(sum_mape/4)
    return list_of_mae, list_of_mape

def plot_all(list_of_next_4_since_current_time, hospitalization, index_2_state):

    list_of_next_4_since_current_time = [np.mean(list_of_next_4_since_current_time[i],axis=0) for i in range(len(list_of_next_4_since_current_time))]
    list_of_next_4_since_current_time = np.mean(list_of_next_4_since_current_time, axis=0)
    for i in range(hospitalization.shape[1]):
        x_range = range(hospitalization.shape[0]-10+1,hospitalization.shape[0]+1)
        plt.plot(x_range,hospitalization[-10:,i], color = 'blue', marker = '>', markersize = 2.0, label ='ground truth')
        prev_6_timestamps = hospitalization[-10:-4,i]
        next_4_timestamps = list_of_next_4_since_current_time[i]
        x_value = np.concatenate((prev_6_timestamps, next_4_timestamps), axis=None)
        plt.plot(x_range[5:], x_value[5:], color='red', linestyle='--', marker='o',
                     markersize=2.0, label = 'prediction')
        plt.title('{} prediction'.format(index_2_state[i]))
        plt.xlabel('Time (Weeks)')
        plt.ylabel('# Hospitalizations')
        plt.legend()
        plt.show()




if __name__ == "__main__":
    exp_name_ls = ['geo','puredata','MTE_based','MTE_static']
    best_epoch_20_runs = list()
    best_20_runs_train_adj_list = list()
    exp_num = 10
    for exp_name in exp_name_ls:
        main(exp_name,exp_num, best_20_runs_train_adj_list, best_epoch_20_runs)





