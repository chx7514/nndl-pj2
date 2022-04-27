import os
from slbi_toolbox import SLBI_ToolBox
import torch
from utils import *
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import lenet
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
torch.backends.cudnn.benchmark = True
load_pth = torch.load('lenet.pth')
load_pth2 = torch.load('lenet_sgd.pth')
torch.cuda.empty_cache()
model = lenet.Net().cuda()
model2 = lenet.Net().cuda()
model.load_state_dict(load_pth['model'])
model2.load_state_dict(load_pth2['model'])

test_loader = torch.utils.data.DataLoader(datasets.MNIST('./data/MNIST', train=False, download=False, transform=transforms.Compose([transforms.ToTensor(),])), batch_size=128, shuffle=False)
device = torch.device("cuda" if (torch.cuda.is_available()) else "cpu")

def get_accuracy(model, test_loader):
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

print('slbi\taccuracy:{0:.4f}'.format(get_accuracy(model, test_loader)))
print('sgd\taccuracy:{0:.4f}'.format(get_accuracy(model2, test_loader)))

weight = model.conv3.weight.clone().detach().cpu().numpy()
weight2 = model2.conv3.weight.clone().detach().cpu().numpy()

H = 16
W = 16

slbi_weight = np.zeros((H * 5, W * 5))
for i in range(H):
    for j in range(W):
        slbi_weight[i*5:i*5+5, j*5:j*5+5] = weight[i][j]
slbi_weight = np.abs(slbi_weight)

sgd_weight = np.zeros((H * 5, W * 5))
for i in range(H):
    for j in range(W):
        sgd_weight[i*5:i*5+5, j*5:j*5+5] = weight2[i][j]
sgd_weight = np.abs(sgd_weight)

plt.figure()
plt.clf()
plt.subplot(1,2,1)
plt.imshow(sgd_weight,cmap='gray')
plt.axis('off')
plt.title('sgd')

plt.subplot(1,2,2)
plt.imshow(slbi_weight,cmap='gray')
plt.axis('off')
plt.title('slbi')

plt.savefig('cmp_weight_sparsity_lenet.png')