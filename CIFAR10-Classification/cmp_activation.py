from multiprocessing import reduction
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
import torch.nn.functional as F
# from torchvision.models.resnet import resnet18
# from torchvision.models.vgg import vgg11
from tqdm import tqdm
import numpy as np

import errno
import os
import os.path as osp
import shutil
from collections import OrderedDict
import time

import matplotlib.pyplot as plt


mean = [0.4913997551666284, 0.48215855929893703, 0.4465309133731618]
std = [0.2023, 0.1994, 0.2010]

train_transform = transforms.Compose([
            transforms.RandomCrop(size=(32, 32), padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

val_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

train_dataset = CIFAR10(root='/root/Desktop/cifar10', train=True, download=False, transform=train_transform)
valid_dataset = CIFAR10(root='/root/Desktop/cifar10', train=False, download=False, transform=val_transform)

Batch_size = 128
train_loader = DataLoader(train_dataset,
                              batch_size=Batch_size,
                              shuffle=True,
                              num_workers=2)
valid_loader = DataLoader(valid_dataset,
                            batch_size=Batch_size,
                            num_workers=2)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    #滑动平均
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    #topk准确率
    #预测结果前k个中出现的正确结果的次数
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def mkdir_if_missing(directory):
    #创建文件夹，如果这个文件夹不存在的话
    if not osp.exists(directory):
        try:
            os.makedirs(directory)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

def save_checkpoint(state, is_best=False, fpath=''):
    if len(osp.dirname(fpath)) != 0:
        mkdir_if_missing(osp.dirname(fpath))
    torch.save(state, fpath)
    if is_best:
        shutil.copy(fpath, osp.join(osp.dirname(fpath), 'best_model.pth.tar'))

def warp_tqdm(data_loader, disable_tqdm):
    #进度条打印
    if disable_tqdm:
        tqdm_loader = data_loader
    else:
        tqdm_loader = tqdm(data_loader, ncols=0)
    return tqdm_loader

def train(train_loader, model, criterion, optimizer, epoch):
    #每个epoch的优化过程
    losses = AverageMeter()

    # switch to train mode
    model.train()

    for input, target in warp_tqdm(train_loader, True):


        input = input.cuda()
        target = target.cuda()

        # compute output
        output = model(input)
        loss = criterion(output, target)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure accuracy and record loss
        losses.update(loss.item(), input.size(0))

    log = 'Epoch:{0}\tLoss: {loss.avg:.4f}\t'.format(epoch, loss=losses)
    return losses.avg, log


def test(test_loader, model, criterion):
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    for input, target in test_loader:

        # compute output
        with torch.no_grad():
            input = input.cuda()
            target = target.cuda()
            output = model(input)

        # measure accuracy and record loss
        acc1 = accuracy(output.data, target)[0]
        top1.update(acc1.item(), input.size(0))

        # measure elapsed time

    log = 'Test Acc@1: {top1.avg:.3f}'.format(top1=top1)

    return top1.avg, log

num_epochs = 20

from net.convnet import ConvNet

# ReLU
model = ConvNet()
model = model.cuda()
train_loader = train_loader
test_loader = valid_loader
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9, weight_decay=5e-4)
criterion = nn.CrossEntropyLoss()
best_acc = 0
relu_accs = []
relu_loss = []
T1 = time.clock()
for epoch in range(num_epochs):
    loss, train_log = train(train_loader, model, criterion, optimizer, epoch)
    acc, test_log = test(test_loader, model, criterion)
    log = train_log + test_log
    print(log)
    is_best = acc > best_acc
    best_acc = max(acc, best_acc)
    relu_accs.append(acc)
    relu_loss.append(loss)
T2 = time.clock()
print('training time:{0:.2f}s'.format(T2 - T1))

# Sigmoid
model = ConvNet(activation=nn.Sigmoid())
model = model.cuda()
train_loader = train_loader
test_loader = valid_loader
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9, weight_decay=5e-4)
criterion = nn.CrossEntropyLoss()
best_acc = 0
Sigmoid_accs = []
Sigmoid_loss = []
T1 = time.clock()
for epoch in range(num_epochs):
    loss, train_log = train(train_loader, model, criterion, optimizer, epoch)
    acc, test_log = test(test_loader, model, criterion)
    log = train_log + test_log
    print(log)
    is_best = acc > best_acc
    best_acc = max(acc, best_acc)
    Sigmoid_accs.append(acc)
    Sigmoid_loss.append(loss)
T2 = time.clock()
print('training time:{0:.2f}s'.format(T2 - T1))

# tanh
model = ConvNet(activation=nn.Tanh())
model = model.cuda()
train_loader = train_loader
test_loader = valid_loader
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9, weight_decay=5e-4)
criterion = nn.CrossEntropyLoss()
best_acc = 0
tanh_accs = []
tanh_loss = []
T1 = time.clock()
for epoch in range(num_epochs):
    loss, train_log = train(train_loader, model, criterion, optimizer, epoch)
    acc, test_log = test(test_loader, model, criterion)
    log = train_log + test_log
    print(log)
    is_best = acc > best_acc
    best_acc = max(acc, best_acc)
    tanh_accs.append(acc)
    tanh_loss.append(loss)
T2 = time.clock()
print('training time:{0:.2f}s'.format(T2 - T1))

# elu
model = ConvNet(activation=nn.ELU(inplace=True))
model = model.cuda()
train_loader = train_loader
test_loader = valid_loader
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9, weight_decay=5e-4)
criterion = nn.CrossEntropyLoss()
best_acc = 0
elu_accs = []
elu_loss = []
T1 = time.clock()
for epoch in range(num_epochs):
    loss, train_log = train(train_loader, model, criterion, optimizer, epoch)
    acc, test_log = test(test_loader, model, criterion)
    log = train_log + test_log
    print(log)
    is_best = acc > best_acc
    best_acc = max(acc, best_acc)
    elu_accs.append(acc)
    elu_loss.append(loss)
T2 = time.clock()
print('training time:{0:.2f}s'.format(T2 - T1))

# LeakyReLU
model = ConvNet(activation=nn.LeakyReLU(inplace=True))
model = model.cuda()
train_loader = train_loader
test_loader = valid_loader
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9, weight_decay=5e-4)
criterion = nn.CrossEntropyLoss()
best_acc = 0
LeakyReLU_accs = []
LeakyReLU_loss = []
T1 = time.clock()
for epoch in range(num_epochs):
    loss, train_log = train(train_loader, model, criterion, optimizer, epoch)
    acc, test_log = test(test_loader, model, criterion)
    log = train_log + test_log
    print(log)
    is_best = acc > best_acc
    best_acc = max(acc, best_acc)
    LeakyReLU_accs.append(acc)
    LeakyReLU_loss.append(loss)
T2 = time.clock()
print('training time:{0:.2f}s'.format(T2 - T1))

plt.subplot(1,2,1)
plt.plot(relu_loss,label='ReLU')
plt.plot(Sigmoid_loss,label='Sigmoid')
plt.plot(tanh_loss,label='Tanh')
plt.plot(elu_loss,label='ELU')
plt.plot(LeakyReLU_loss,label='LeakyReLU')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend()

plt.subplot(1,2,2)
plt.plot(relu_accs,label='ReLU')
plt.plot(Sigmoid_accs,label='Sigmoid')
plt.plot(tanh_accs,label='Tanh')
plt.plot(elu_accs,label='ELU')
plt.plot(LeakyReLU_accs,label='LeakyReLU')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.legend()
plt.savefig('compare_activation.png')