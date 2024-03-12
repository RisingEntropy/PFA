import torch
import torch.nn as nn
from prefetch_generator import BackgroundGenerator
from torch.utils.data import DataLoader

import VGG
from Experiment import Argument, DVSBasedExperiment
from spikingjelly.datasets.cifar10_dvs import CIFAR10DVS


DVSBasedExperiment.setup_Environment()
cifar10dvs_arg = Argument()
cifar10dvs_arg.device = "cuda:0"
cifar10dvs_arg.datasetdir = "/Datasets/CIFAR10DVS"
cifar10dvs_arg.T = 10
cifar10dvs_arg.rank = 16
cifar10dvs_arg.b = 32
cifar10dvs_arg.epoch = 200
cifar10dvs_arg.num_class = 10
experiment = DVSBasedExperiment("CIFAR10DVS_T14_r8", cifar10dvs_arg)
net = VGG.VGG(num_class=cifar10dvs_arg.num_class,T=cifar10dvs_arg.T, rank=cifar10dvs_arg.rank, device=cifar10dvs_arg.device).to(cifar10dvs_arg.device)
experiment.train(net, CIFAR10DVS, cifar10dvs_arg.num_class, train_set=None, validate_set=None)


# For NCaltech-101
# from spikingjelly.datasets.n_caltech101 import NCaltech101
# experiment.train(net, NCaltech101, 101, train_set=None, validate_set=None)

# For datasets without a wrapping class, please use code below:
# train_set = DVS128Gesture(cifar10dvs_arg.datasetdir, train=True, data_type='frame', split_by='number',
#                               frames_number=cifar10dvs_arg.T)
# test_set = DVS128Gesture(cifar10dvs_arg.datasetdir, train=False, data_type='frame', split_by='number',
#                              frames_number=cifar10dvs_arg.T)
# experiment.train(net, None, 110, train_set=train_set, validate_set=test_set)

