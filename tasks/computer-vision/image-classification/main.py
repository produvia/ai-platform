"""
Download and create a YOLOv3 Keras model and save it to file and 
load yolov3 model and perform object detection.

Edited from the ai-platform object detection. Allows for image, video and passing of parameters from the main to subsequent entry points
"""

from __future__ import print_function
import argparse
import os
import mlflow
import tempfile
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from tensorboardX import SummaryWriter


# Command-line arguments
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--enable-cuda', type=str, choices=['True', 'False'], default='True',
                    help='enables or disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='how many batches to wait before logging training status')
args = parser.parse_args()


cuda = True if args.enable_cuda == 'True' else False

args.cuda = cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)

if args.cuda:
	torch.cuda.manual_seed(args.seed)

cifar_train_mean = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
cifar_train_std = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)
train_loader = torch.utils.data.DataLoader(
	datasets.CIFAR('../data', train=True, download=True,
					transform=transforms.Compose([
					transforms.toTensor(),
					transforms.Normalize(mean=cifar_train_mean, std=cifar_train_std)
					])),
	batch_size=args.batch_size, shuffle=True, pin_memory=True)


test_loader = torch.utils.data.DataLoader(
	datasets.CIFAR('../data', train=False, download=True,
					transform=transforms.Compose([
					transforms.toTensor(),
					transforms.Normalize(cifar_train_mean,cifar_train_std)
					])),
	batch_size=args.batch_size, shuffle=True, pin_memory=True)


