import os
GOOGLE_DRIVE_PATH_AFTER_MYDRIVE = None
# GOOGLE_DRIVE_PATH = 'D:\Study\cs231n\A3'
GOOGLE_DRIVE_PATH = '/home/xjk/cs231n/A3'
import sys
sys.path.append(GOOGLE_DRIVE_PATH)

import time, os
os.environ["TZ"] = "US/Eastern"
# time.tzset()

from convolutional_networks import hello_convolutional_networks

from a3_helper import hello_helper

convolutional_networks_path = os.path.join(GOOGLE_DRIVE_PATH, 'convolutional_networks.py')
convolutional_networks_edit_time = time.ctime(os.path.getmtime(convolutional_networks_path))

import eecs598
import torch
import torchvision
import matplotlib.pyplot as plt
import statistics
import random
import time
import math

from eecs598 import reset_seed, Solver

import eecs598

eecs598.reset_seed(0)
data_dict = eecs598.data.preprocess_cifar10(cuda=True, show_examples=False, dtype=torch.float64, flatten=False)
print('Train data shape: ', data_dict['X_train'].shape)
print('Train labels shape: ', data_dict['y_train'].shape)
print('Validation data shape: ', data_dict['X_val'].shape)
print('Validation labels shape: ', data_dict['y_val'].shape)
print('Test data shape: ', data_dict['X_test'].shape)
print('Test labels shape: ', data_dict['y_test'].shape)

from convolutional_networks import DeepConvNet
from fully_connected_networks import sgd_momentum, adam
reset_seed(0)

# # Try training a very deep net with batchnorm
num_train = 10000
small_data = {
  'X_train': data_dict['X_train'][:num_train],
  'y_train': data_dict['y_train'][:num_train],
  'X_val': data_dict['X_val'],
  'y_val': data_dict['y_val'],
}
input_dims = data_dict['X_train'].shape[1:]
num_epochs = 5
lrs = [2e-1, 1e-1, 5e-2]
lrs = [5e-3, 1e-2, 2e-2]

solvers = []
for lr in lrs:
  print('No normalization: learning rate = ', lr)
  model = DeepConvNet(input_dims=input_dims, num_classes=10,
                      # num_filters=[8, 8, 8],
                      # max_pools=[0, 1, 2],
                      num_filters=[8, 8, 16, 16, 32, 32],
                      max_pools=[1, 3, 5],
                      weight_scale='kaiming',
                      batchnorm=True,
                      reg=1e-5, dtype=torch.float32, device='cuda')
  solver = Solver(model, small_data,
                  num_epochs=num_epochs,
                  # batch_size=100,
                  batch_size=128,
                  update_rule=sgd_momentum,
                  optim_config={
                    'learning_rate': lr,
                  },
                  verbose=False, device='cuda')
  solver.train()
  solvers.append(solver)

# bn_solvers = []
# for lr in lrs:
#   print('Normalization: learning rate = ', lr)
#   bn_model = DeepConvNet(input_dims=input_dims, num_classes=10,
#                          num_filters=[8, 8, 16, 16, 32, 32],
#                          max_pools=[1, 3, 5],
#                          weight_scale='kaiming',
#                          batchnorm=True,
#                          reg=1e-5, dtype=torch.float32, device='cpu')
#   bn_solver = Solver(bn_model, small_data,
#                      num_epochs=num_epochs, batch_size=128,
#                      update_rule=sgd_momentum,
#                      optim_config={
#                        'learning_rate': lr,
#                      },
#                      verbose=False, device='cpu')
#   bn_solver.train()
#   bn_solvers.append(bn_solver)
