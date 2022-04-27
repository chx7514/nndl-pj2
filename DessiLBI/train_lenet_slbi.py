from cgi import test
import os
from slbi_toolbox import SLBI_ToolBox
import torch
from utils import *
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import argparse
import lenet
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

torch.backends.cudnn.benchmark = True
parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=128, type=int)
parser.add_argument("--lr", default=1e-1, type=float)
parser.add_argument("--interval", default=20, type=int)
parser.add_argument("--kappa", default=1, type=int)
parser.add_argument("--dataset", default='MNIST', type=str)
parser.add_argument("--train", default=True, type=str2bool)
parser.add_argument("--download", default=True, type=str2bool)
parser.add_argument("--shuffle", default=True, type=str2bool)
parser.add_argument("--use_cuda", default=True, type=str2bool)
parser.add_argument("--parallel", default=False, type=str2bool)
parser.add_argument("--epoch", default=60, type=int)
parser.add_argument("--model_name", default='lenet', type=str)
parser.add_argument("--gpu_num", default='0', type=str)
parser.add_argument("--mu", default=20, type=int)
args = parser.parse_args()
name_list = []
device = torch.device("cuda" if (args.use_cuda and torch.cuda.is_available()) else "cpu")
torch.cuda.empty_cache() 
model = lenet.Net().to(device)
if args.parallel:
    model = nn.DataParallel(model)
else:
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_num
layer_list = []
for name, p in model.named_parameters():
    name_list.append(name)
    print(name)
    if len(p.data.size()) == 4 or len(p.data.size()) == 2:
        layer_list.append(name)
optimizer = SLBI_ToolBox(model.parameters(), lr=args.lr, kappa=args.kappa, mu=args.mu, weight_decay=0)
optimizer.assign_name(name_list)
optimizer.initialize_slbi(layer_list)
train_loader = torch.utils.data.DataLoader(datasets.MNIST('./data/MNIST', train=True, download=False, transform=transforms.Compose([transforms.ToTensor(),])), batch_size=args.batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(datasets.MNIST('./data/MNIST', train=False, download=False, transform=transforms.Compose([transforms.ToTensor(),])), batch_size=args.batch_size, shuffle=False)
all_num = args.epoch * len(train_loader)
print('num of all step:', all_num)
print('num of step per epoch:', len(train_loader))
train_acc = []
test_acc = []
exp_path = 'acc_curve'

def get_accuracy(test_loader):
    model.eval()
    correct = 0
    num = 0
    for pack in test_loader:
        data, target = pack[0].to(device), pack[1].to(device)
        logits = model(data)
        _, pred = logits.max(1)
        correct += pred.eq(target).sum().item()
        num += data.shape[0]
    acc = correct / num 
    return acc 

for ep in range(args.epoch):
    model.train()
    descent_lr(args.lr, ep, optimizer, args.interval)
    for pack in train_loader:
        data, target = pack[0].to(device), pack[1].to(device)
        logits = model(data)
        loss = F.nll_loss(logits, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    train_acc.append(get_accuracy(train_loader))
    test_acc.append(get_accuracy(test_loader))
    optimizer.update_prune_order(ep)
save_model_and_optimizer(model, optimizer, 'lenet.pth')

plt.figure()
plt.clf()
plt.plot(train_acc, label='train')
plt.plot(test_acc, label='test')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.legend()
plt.savefig(exp_path)


print('origin acc:',get_accuracy(test_loader) )
optimizer.prune_layer_by_order_by_name(80, 'conv3.weight', True)
print('acc after prun conv3:',get_accuracy(test_loader))
optimizer.recover()
print('acc after recover:',get_accuracy(test_loader))

print('origin acc:', get_accuracy(test_loader))
optimizer.prune_layer_by_order_by_list(80, ['conv3.weight','fc1.weight'], True)
print('acc after prun conv3 and fc1:',get_accuracy(test_loader))
optimizer.recover()
print('acc after recover:',get_accuracy(test_loader))
