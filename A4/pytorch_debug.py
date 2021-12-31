import os
GOOGLE_DRIVE_PATH = '/home/xjk/cs231n/A4'
import sys
sys.path.append(GOOGLE_DRIVE_PATH)

import time, os
os.environ["TZ"] = "US/Eastern"

from pytorch_autograd_and_nn import *
from a4_helper import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from eecs598.utils import reset_seed
from collections import OrderedDict

# for plotting
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

to_float= torch.float
to_long = torch.long

if torch.cuda.is_available:
  print('Good to go!')
else:
  print('Please set GPU via Edit -> Notebook Settings.')
loader_train, loader_val, loader_test = load_CIFAR(path='./datasets/')

# example of specifications
networks = {
  'plain32': {
    'block': PlainBlock,
    'stage_args': [
      (8, 8, 5, False),
      (8, 16, 5, True),
      (16, 32, 5, True),
    ]
  },
  'resnet32': {
    'block': ResidualBlock,
    'stage_args': [
      (8, 8, 5, False),
      (8, 16, 5, True),
      (16, 32, 5, True),
    ]
  },
}
def check_accuracy_part34(loader, model):
  if loader.dataset.train:
    print('Checking accuracy on validation set')
  else:
    print('Checking accuracy on test set')
  num_correct = 0
  num_samples = 0
  model.eval()  # set model to evaluation mode
  with torch.no_grad():
    for x, y in loader:
      x = x.to(device='cuda', dtype=to_float)  # move to device, e.g. GPU
      y = y.to(device='cuda', dtype=to_long)
      scores = model(x)
      _, preds = scores.max(1)
      num_correct += (preds == y).sum()
      num_samples += preds.size(0)
    acc = float(num_correct) / num_samples
    print('Got %d / %d correct (%.2f)' % (num_correct, num_samples, 100 * acc))
  return acc


def adjust_learning_rate(optimizer, lrd, epoch, schedule):
    """
    Multiply lrd to the learning rate if epoch is in schedule

    Inputs:
    - optimizer: An Optimizer object we will use to train the model
    - lrd: learning rate decay; a factor multiplied at scheduled epochs
    - epochs: the current epoch number
    - schedule: the list of epochs that requires learning rate update

    Returns: Nothing, but learning rate might be updated
    """
    if epoch in schedule:
        for param_group in optimizer.param_groups:
            print('lr decay from {} to {}'.format(param_group['lr'], param_group['lr'] * lrd))
            param_group['lr'] *= lrd

def train_part345(model, optimizer, epochs=1, learning_rate_decay=.1, schedule=[], verbose=True):
    """
    Train a model on CIFAR-10 using the PyTorch Module API.

    Inputs:
    - model: A PyTorch Module giving the model to train.
    - optimizer: An Optimizer object we will use to train the model
    - epochs: (Optional) A Python integer giving the number of epochs to train for

    Returns: Nothing, but prints model accuracies during training.
    """
    model = model.to(device='cuda')  # move the model parameters to CPU/GPU
    num_iters = epochs * len(loader_train)
    print_every = 100
    if verbose:
        num_prints = num_iters // print_every + 1
    else:
        num_prints = epochs
    acc_history = torch.zeros(num_prints, dtype=to_float)
    iter_history = torch.zeros(num_prints, dtype=to_long)
    for e in range(epochs):

        adjust_learning_rate(optimizer, learning_rate_decay, e, schedule)

        for t, (x, y) in enumerate(loader_train):
            model.train()  # put model to training mode
            x = x.to(device='cuda', dtype=to_float)  # move to device, e.g. GPU
            y = y.to(device='cuda', dtype=to_long)

            scores = model(x)
            loss = F.cross_entropy(scores, y)

            # Zero out all of the gradients for the variables which the optimizer
            # will update.
            optimizer.zero_grad()

            # This is the backwards pass: compute the gradient of the loss with
            # respect to each  parameter of the model.
            loss.backward()

            # Actually update the parameters of the model using the gradients
            # computed by the backwards pass.
            optimizer.step()

            tt = t + e * len(loader_train)

            if verbose and (tt % print_every == 0 or (e == epochs - 1 and t == len(loader_train) - 1)):
                print('Epoch %d, Iteration %d, loss = %.4f' % (e, tt, loss.item()))
                acc = check_accuracy_part34(loader_val, model)
                acc_history[tt // print_every] = acc
                iter_history[tt // print_every] = tt
                print()
            elif not verbose and (t == len(loader_train) - 1):
                print('Epoch %d, Iteration %d, loss = %.4f' % (e, tt, loss.item()))
                acc = check_accuracy_part34(loader_val, model)
                acc_history[e] = acc
                iter_history[e] = tt
                print()
    return acc_history, iter_history


def get_resnet(name):
  # YOUR_TURN: Impelement ResNet.__init__ and ResNet.forward
  return ResNet(**networks[name])


names = ['plain32', 'resnet32']
acc_history_dict = {}
iter_history_dict = {}
for name in names:
    reset_seed(0)
    print(name, '\n')
    model = get_resnet(name)
    #   init_module(model)

    optimizer = optim.SGD(model.parameters(), lr=1e-2, momentum=.9, weight_decay=1e-4)

    acc_history, iter_history = train_part345(model, optimizer, epochs=10, schedule=[6, 8], verbose=False)
    acc_history_dict[name] = acc_history
    iter_history_dict[name] = iter_history