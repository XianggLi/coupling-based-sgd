import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse

from models import *
from utils import progress_bar

import math
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim.lr_scheduler import _LRScheduler

import matplotlib.pyplot as plt 
from google.colab import files

from psutil import virtual_memory

def set_random_seed(seed):
    random.seed(seed) 
    np.random.seed(seed) 
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True 
    cudnn.benchmark = False

set_random_seed(42)

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  
start_epoch = 0  
test_accuracy_values = []
test_accuracy_values_3 = [] # Store test accuracy for Model 3
test_accuracy_values_4 = [] # Store test accuracy for Model 4
D_k_values = []  

print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=128, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=128, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

print('==> Building model..')

def initialize_close_model(net1, net2, noise_level=0.01):
    net2.load_state_dict(net1.state_dict())

    for param in net2.parameters():
        noise = torch.randn_like(param) * noise_level
        param.data.add_(noise)

    return net2

net1 = ResNet18().to(device)
net2 = ResNet18().to(device)
net3 = ResNet18().to(device) 
net4 = ResNet18().to(device) 

net2 = initialize_close_model(net1, net2)

# Make Model 3 and Model 4 have the same initial parameters as Model 1
net3.load_state_dict(net1.state_dict())  
net4.load_state_dict(net1.state_dict()) 

if device == 'cuda':
    net1 = torch.nn.DataParallel(net1)
    net2 = torch.nn.DataParallel(net2)
    net3 = torch.nn.DataParallel(net3)
    net4 = torch.nn.DataParallel(net4)
    cudnn.benchmark = True

# if args.resume:
#     print('==> Resuming from checkpoint..')
#     assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
#     checkpoint = torch.load('./checkpoint/ckpt.pth')
#     net.load_state_dict(checkpoint['net'])
#     best_acc = checkpoint['acc']
#     start_epoch = checkpoint['epoch']

criterion = nn.CrossEntropyLoss()
optimizer1 = optim.SGD(net1.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)
optimizer2 = optim.SGD(net2.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)
optimizer3 = optim.SGD(net3.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)
optimizer4 = optim.SGD(net4.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)


class CustomScheduler(_LRScheduler):
    def __init__(self, optimizer, factor, patience, threshold, min_lr=1e-6, verbose=False):
        self.factor = factor
        self.patience = patience
        self.threshold = threshold
        self.min_lr = min_lr
        self.verbose = verbose
        self.best_D_k = 0
        self.num_bad_epochs = 0
        self.optimizer = optimizer
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
        self._last_lr = self.base_lrs
        super(CustomScheduler, self).__init__(optimizer)

    def get_lr(self):
        return [max(base_lr * self.factor ** (self.num_bad_epochs // self.patience), self.min_lr)
                for base_lr in self.base_lrs]

    def step(self, D_k=None, epoch=None):
        if D_k is None:
            super(CustomScheduler, self).step(epoch)
            return

        if D_k > self.best_D_k:
            self.best_D_k = D_k
            self.num_bad_epochs = 0
        elif D_k < self.threshold * self.best_D_k:
            self.num_bad_epochs += 1
            if self.num_bad_epochs >= self.patience:
                self._reduce_lr(epoch)
                self.num_bad_epochs = 0

    def _reduce_lr(self, epoch):
        for i, param_group in enumerate(self.optimizer.param_groups):
            old_lr = float(param_group['lr'])
            new_lr = max(old_lr * self.factor, self.min_lr)
            if old_lr - new_lr > 1e-8:
                param_group['lr'] = new_lr
                self.best_D_k = 0 
                if self.verbose:
                    print(f'Epoch {epoch}: reducing learning rate of group {i} to {new_lr:.4e}. Resetting best_D_k to zero.')

    def get_last_lr(self):
        return self._last_lr


scheduler1 = CustomScheduler(optimizer1, factor=0.1, patience=10, threshold=0.95, verbose=True)
scheduler2 = CustomScheduler(optimizer2, factor=0.1, patience=10, threshold=0.95, verbose=True)

def train(epoch):
    print('\nEpoch: %d' % epoch)

    net1.train()
    net2.train()
    net3.train()
    net4.train()

    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer1.zero_grad()
        outputs1 = net1(inputs)
        loss1 = criterion(outputs1, targets)
        loss1.backward()
        optimizer1.step()

        optimizer2.zero_grad()
        outputs2 = net2(inputs)
        loss2 = criterion(outputs2, targets)
        loss2.backward()
        optimizer2.step()

        optimizer3.zero_grad()
        outputs3 = net3(inputs)
        loss3 = criterion(outputs3, targets)
        loss3.backward()
        optimizer3.step()

        optimizer4.zero_grad()
        outputs4 = net4(inputs)
        loss4 = criterion(outputs4, targets)
        loss4.backward()
        optimizer4.step()

        train_loss += loss1.item()
        _, predicted = outputs1.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        # progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
        #              % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))


def test(epoch, net, test_accuracy_values):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
    test_accuracy_values.append(correct/total)

# Load or calculate results
results_file = 'results_1104.pt'
if os.path.exists(results_file):
    print("Loading results from file...")
    loaded_data = torch.load(results_file)
    test_accuracy_values = loaded_data['test_acc_cb']
    test_accuracy_values_3 = loaded_data['test_acc_constant']
    test_accuracy_values_4 = loaded_data['test_acc_manual']
    D_k_values = loaded_data['D_k_values']

EPOCHS = 200
for epoch in range(start_epoch, start_epoch+EPOCHS):
    train(epoch)
    
    # Test all four models
    test(epoch, net1, test_accuracy_values)
    test(epoch, net3, test_accuracy_values_3)
    test(epoch, net4, test_accuracy_values_4)

    # if epoch == 50:
    #     optimizer3.param_groups[0]['lr'] = optimizer3.param_groups[0]['lr'] * 0.1
    #     print("3rd SGD's learning rate: ", optimizer3.param_groups[0]["lr"])
    
    if epoch == 75:
        optimizer4.param_groups[0]['lr'] = optimizer4.param_groups[0]['lr'] * 0.1
        print("4th SGD's learning rate: ", optimizer4.param_groups[0]["lr"])

    D_k = 0
    for param1, param2 in zip(optimizer1.param_groups[0]['params'], optimizer2.param_groups[0]['params']):
        D_k += torch.norm(param1 - param2, 2)
    D_k_values.append(D_k.item())
    print("Epoch %d, current Dk is %.2f:" % (epoch, D_k.item()))

    if epoch == 170:
        ram_gb = virtual_memory().total / 1e9
        print('Your runtime has {:.1f} gigabytes of available RAM\n'.format(ram_gb))

        if ram_gb < 20:
            print('Not using a high-RAM runtime')
        else:
            print('You are using a high-RAM runtime!')

    before_lr = optimizer1.param_groups[0]["lr"]
    scheduler1.step(D_k.item(), epoch)
    scheduler2.step(D_k.item(), epoch)
    after_lr = optimizer1.param_groups[0]["lr"]
    if after_lr < before_lr:    
        net2 = initialize_close_model(net1, net2)
        print("Epoch %d: proximity-based SGD lr %.4f -> %.4f" % (epoch, before_lr, after_lr))

# Save results
torch.save({
    'test_acc_cb': test_accuracy_values,
    'test_acc_constant': test_accuracy_values_3,
    'test_acc_manual': test_accuracy_values_4,
    'D_k_values': D_k_values
}, results_file)


plt.figure(figsize=(10, 6))
plt.plot(range(start_epoch, start_epoch + EPOCHS), test_accuracy_values, marker='o', color='b', label='Coupling-based SGD')
plt.plot(range(start_epoch, start_epoch + EPOCHS), test_accuracy_values_3, marker='x', color='r', label='Fixed step-size')
plt.plot(range(start_epoch, start_epoch + EPOCHS), test_accuracy_values_4, marker='s', color='g', label='Fixed step-size, \\gamma=0.001')
plt.xlabel('Epoch')
plt.ylabel('Test Accuracy (%)')
# plt.title('proximity_based SGD')
plt.legend()
plt.grid(True)
plt.savefig('proximity_based_all_models_1104.png')

plt.ylim(0.80, 0.96)
# plt.title('proximity_based SGD - Y-axis from 80%')
plt.savefig('proximity_based_all_models_yaxisfrom80_1104.png')

plt.figure(figsize=(10, 6))
plt.plot(range(start_epoch, start_epoch + EPOCHS), D_k_values, marker='o', color='r', label='D_k')
plt.xlabel('Epoch')
plt.ylabel('D_k')
plt.title('Statistic')
plt.legend()
plt.grid(True)
plt.savefig('Dk_0815_lr001_1104.png')

plt.figure(figsize=(12, 6))
plt.plot(range(start_epoch, start_epoch + EPOCHS), test_accuracy_values, color='b', label=r'Coupling-based scheduler')
plt.plot(range(start_epoch, start_epoch + EPOCHS), test_accuracy_values_3, color='r', label=r'Fixed step-size')
plt.plot(range(start_epoch, start_epoch + EPOCHS), test_accuracy_values_4, color='g', label=r'Manual-decrease step-size')
plt.xlabel('Epoch', fontsize=25)
plt.ylabel('Test Accuracy (%)', fontsize=25)
plt.legend(fontsize=18)  # Set legend fontsize here
# plt.legend(loc='best', fontsize=12, frameon=False)  # 'best' automatically places the legend in the optimal location
# plt.title('proximity_based SGD')
plt.grid(True, linestyle='--', alpha=0.7)
# plt.savefig('fig5.png')

plt.ylim(0.80, 0.95)
# plt.title('proximity_based SGD - Y-axis from 80%')
plt.savefig('fig5_1104.png')
