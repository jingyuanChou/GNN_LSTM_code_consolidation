import torch.nn

from util import *
import graph_nets
from rnn import *

# CONSTRUCT MODELS
WSC = WeightedSAGEConv
USC = lambda in_channels, out_channels, bias=True: WeightedSAGEConv(in_channels, out_channels, weighted=False)
linear_module = lambda in_channels, out_channels, bias: graph_nets.GraphLinear(in_channels, out_channels, bias=bias)
DeepWSC = lambda lookback, dim: graph_nets.GNNModule(WSC, 1, lookback, dim=dim, res_factors=[0.5], dropouts=[0.5])

print(DeepWSC)

args = {
  # Number of previous timesteps to use for prediction
  "lookback": 12,

  # Pattern of previous timesteps to use for prediction, use previous weeks data to predict next few timestamps
  "lookback_pattern": [11,10,
                       9,8,7,6,5,4,3,2,1,0],

  # Number of edges in the graph - this was a hotfix, needs to be deleted/resolved
  "edge_count": 0,

  # Number of folds in K-fold cross validation
  "K": 5,

  # Should perform K-fold cross validation instead of normal training
  "CROSS_VALIDATE": False,

  # Threshold for creation of edges based on geodesic distance
  "DISTANCE_THRESHOLD": 250,  # km

  # Minimum number of edges per node in graph
  "EDGES_PER_NODE": 3,

  # Name of loss function to be used in training
  "loss_func": mase2_loss,

  # Name of reporting metric
  "reporting_metric": MAE_loss,

  # A description to be included in the results output file
  "experiment_description": "GraphLSTM Ablation FINAL",

  # Number of epochs to train models
  "num_epochs": 30,

  # Name of optimizer used in training
  "optimizer": torch.optim.Adam,

  # Learning rate of optimizer used in training
  "learning_rate": 0.1,

  # Percentage of dataset to use for training (less than 1.0 to speed up training)
  "sample": 1.0,

  # Features to train on (of the nodes)
  "features": ["new_cases_smoothed"],

  # Edge features to train on
  "mobility_edge_features": [
                    # "distance",  # geodesic distance between land mass centroids of countries
                    # "flights",   # number of flights between countries
                     "sci"  # Facebook Social Connectivity Index
  ],

  "models": []
}

models = [
    RNN(module=WSC, gnn=DeepWSC, rnn=LSTM, dim=4, gnn_2=None, rnn_depth=1, name="Our Model", node_features=1,output = 1,
        skip_connection=True),
    #graph_nets.RecurrentGraphNet(GConvLSTM),
    # graph_nets.RecurrentGraphNet(GConvGRU),
    # graph_nets.RecurrentGraphNet(DCRNN),
    # graph_nets.RecurrentGraphNet(GCLSTM),
    # graph_nets.LagPredictor()
    ]
'''
models = [
    RNN(module=WSC, gnn=None, rnn=LSTM, dim=16, gnn_2=None, rnn_depth=1, name="Our Model", node_features=1,output = 1,
        skip_connection=True)
]
'''

args['models'] = models

class Parameters:
    def __init__(self):
        # parser = argparse.ArgumentParser('Recurrent GNN COVID Prediction')
        #
        # try:
        #     args = parser.parse_args()
        #     with open('parameters.json', 'rt') as f:
        #         t_args = argparse.Namespace()
        #         t_args.__dict__.update(json.load(f))
        #         args = parser.parse_args(namespace=t_args)
        # except:
        #     parser.print_help()

        self.lookback  = args['lookback']
        self.lookback_pattern = args['lookback_pattern']
        self.edge_count = args['edge_count']
        self.K = args['K']
        self.CROSS_VALIDATE = args['CROSS_VALIDATE']
        self.DISTANCE_THRESHOLD = args['DISTANCE_THRESHOLD']
        self.EDGES_PER_NODE = args['EDGES_PER_NODE']
        self.experiment_description = args['experiment_description']
        self.num_epochs = args['num_epochs']
        self.learning_rate = args['learning_rate']
        self.sample = args['sample']
        self.features = args['features']
        self.mobility_edge_features = args['mobility_edge_features']

        self.loss_func = args['loss_func']
        self.reporting_metric = args['reporting_metric']
        self.optimizer = args['optimizer']
        self.models = args['models']
    
    def get_optimizer(self, model_params):
        return self.optimizer(model_params, self.learning_rate)


# loss_func/reporting_metric, models, and optimizer (get_optimizer()?) need special initializing or maybe you could change the json to a python file
