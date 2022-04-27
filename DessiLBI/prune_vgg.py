import os
from slbi_toolbox_adam import SLBI_ToolBox
import torch
from utils import *
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import vgg
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
torch.backends.cudnn.benchmark = True
load_pth = torch.load('vgg_adam.pth')
torch.cuda.empty_cache()
model = vgg.VGG_A().cuda()
model.load_state_dict(load_pth['model'])
name_list = []
layer_list = []
for name, p in model.named_parameters():
    name_list.append(name)
    if len(p.data.size()) == 4 or len(p.data.size()) == 2:
        layer_list.append(name)
optimizer = SLBI_ToolBox(model.parameters(), lr=1e-3, kappa=1, mu=20, weight_decay=0)
optimizer.load_state_dict(load_pth['optimizer'])
optimizer.assign_name(name_list)
optimizer.initialize_slbi(layer_list)
normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                     std=[0.5, 0.5, 0.5])
test_loader = torch.utils.data.DataLoader(datasets.CIFAR10('./cifar10', train=False, download=False, transform=transforms.Compose([transforms.ToTensor(), normalize])), batch_size=128, shuffle=False)
device = torch.device("cuda" if (torch.cuda.is_available()) else "cpu")

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

# prune the third conv layer
def prune_result(ratio):
    optimizer.prune_layer_by_order_by_name(ratio, 'conv3_1.weight', True)    
    prun_acc = get_accuracy(test_loader)
    optimizer.recover()
    return prun_acc


ratios = [5, 10, 15, 20, 25, 30]

print('original accuracy:{0:.4f}'.format(get_accuracy(test_loader)))

for ratio in ratios:
    prun_acc = prune_result(ratio)
    print("ratio:{0}\tpruned accuracy:{1:.4f}".format(ratio, prun_acc))
