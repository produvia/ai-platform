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
from vgg import *

# Command-line arguments
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=150, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                    help='learning rate (default: 0.1)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.9)')
parser.add_argument('--weight-decay', type=float, default=1e-4, metavar='Y',
                    help='SGD weight decay (default: 1e-4)')
parser.add_argument('--enable-cuda', type=str, choices=['True', 'False'], default='True',
                    help='enables or disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='how many batches to wait before logging training status')
args = parser.parse_args()

if not os.path.isdir('models'):
    os.makedirs('models')

cuda = True if args.enable_cuda == 'True' else False

args.cuda = cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)

if args.cuda:
    torch.cuda.manual_seed(args.seed)

cifar_train_mean = (0.4914, 0.4822, 0.4465)
cifar_train_std = (0.2023, 0.1994, 0.2010)
train_loader = torch.utils.data.DataLoader(
    datasets.CIFAR100('./data', train=True, download=True,
    				transform=transforms.Compose([
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
    				transforms.Normalize(mean=cifar_train_mean, std=cifar_train_std)
    				])),
    batch_size=args.batch_size, shuffle=True, pin_memory=True)


test_loader = torch.utils.data.DataLoader(
    datasets.CIFAR100('./data', train=False, download=True,
    				transform=transforms.Compose([
                    transforms.ToTensor(),
    				transforms.Normalize(cifar_train_mean,cifar_train_std)
    				])),
    batch_size=args.batch_size, shuffle=True, pin_memory=True)

model = VGG('VGG11')

if args.cuda:
    model.cuda()

optimizer = optim.SGD(model.parameters(), lr=args.lr,
        momentum=args.momentum,weight_decay=args.weight_decay)

writer = None

def train(epoch,best_loss=None):
    model.train()
    total_loss = 0.
    for batch_idx, (data, target) in enumerate(train_loader):
        if args.cuda:
            data,target = data.cuda(),target.cuda()
        data,target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        total_loss += loss.data.item()
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data.item()))
            step = epoch * len(train_loader) + batch_idx
            log_scalar('train_loss', loss.data.item(), step)
            #model.log_weights(step)
    if not best_loss or total_loss < best_loss:
        best_loss = total_loss
        print('Saving current best model....')
        torch.save(model.state_dict(), 'models/cifar100_model.h5')

    return best_loss

def test(epoch):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            if args.cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').data.item() # sum up batch loss
            pred = output.data.max(1)[1] # get the index of the max log-probability
            correct += pred.eq(target.data).cpu().sum().item()

    test_loss /= len(test_loader.dataset)
    test_accuracy = 100.0 * correct / len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset), test_accuracy))
    step = (epoch + 1) * len(train_loader)
    log_scalar('test_loss', test_loss, step)
    log_scalar('test_accuracy', test_accuracy, step)
    return test_accuracy

def log_scalar(name, value, step):
    """Log a scalar value to both MLflow and TensorBoard"""
    writer.add_scalar(name, value, step)
    mlflow.log_metric(name, value)

with mlflow.start_run():
    # Log our parameters into mlflow
    for key, value in vars(args).items():
        mlflow.log_param(key, value)
    
    # Create a SummaryWriter to write TensorBoard events locally
    output_dir = dirpath = tempfile.mkdtemp()
    writer = SummaryWriter(output_dir)
    print("Writing TensorBoard events locally to %s\n" % output_dir)

    # Perform the training
    best_loss = None
    best_acc = 0.
    for epoch in range(1, args.epochs + 1):
        loss = train(epoch, best_loss)
        acc = test(epoch)
        if not best_loss:
            best_loss = loss
        elif best_loss > loss:
            best_loss = loss
            best_acc = acc
    
    print('Training Done. Accuracy = %f'%(best_acc))
    print('Best model saved in models/')
    # Upload the TensorBoard event logs as a run artifact
    print("Uploading TensorBoard events as a run artifact...")
    mlflow.log_artifacts(output_dir, artifact_path="events")
    print("\nLaunch TensorBoard with:\n\ntensorboard --logdir=%s" %
        os.path.join(mlflow.get_artifact_uri(), "events"))
