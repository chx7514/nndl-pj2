import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from torch import nn
import numpy as np
import torch
import os
import random
from tqdm import tqdm as tqdm
from IPython import display

from models.vgg import VGG_A
from models.vgg import VGG_A_BatchNorm
from data.loaders import get_cifar_loader
from utils.nn import set_vgg_weights, get_vgg_weights, set_vgg_gradients, get_vgg_gradients

# ## Constants (parameters) initialization
device_id = [0,1,2,3]
num_workers = 4
batch_size = 128

# add our package dir to path 
module_path = os.path.dirname(os.getcwd())
home_path = module_path
figures_path = os.path.join(home_path, 'reports', 'figures')
models_path = os.path.join(home_path, 'reports', 'models')

# Make sure you are using the right device.
# device_id = device_id
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# device = torch.device("cuda:{}".format(3) if torch.cuda.is_available() else "cpu")
# print(device)
# print(torch.cuda.get_device_name(3))

device = torch.device("cuda:0")



# Initialize your data loader and
# make sure that dataloader works
# as expected by observing one
# sample from it.
train_loader = get_cifar_loader(root='./cifar10', train=True)
val_loader = get_cifar_loader(root='./cifar10', train=False)
# for X,y in train_loader:
#     ## --------------------
#     # Add code as needed
#     #
#     #
#     #
#     #
#     ## --------------------
#     break



# This function is used to calculate the accuracy of model classification
def get_accuracy(test_loader, model):
    size = 0
    correct = 0
    model.eval()
    
    for input, target in test_loader:
        with torch.no_grad():
            input = input.cuda()
            target = target.cuda()
            output = model(input)

        _, y_pred = torch.max(output.data, 1)
        size += target.size(0)
        correct += (y_pred == target).sum().item()
    
    return correct / size

    

# Set a random seed to ensure reproducible results
def set_random_seeds(seed_value=0, device='cpu'):
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    random.seed(seed_value)
    if device != 'cpu': 
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# def VGG_Grad_Pred(model, last_grad):
#     grad = torch.tensor([]).to(device)

#     for layer in model.features:
#         if isinstance(layer, nn.Conv2d):
#             grad = torch.cat((grad, layer.weight.grad.view(-1)))
    
#     for layer in model.classifier:
#         if isinstance(layer, nn.Linear):
#             grad = torch.cat((grad, layer.weight.grad.view(-1)))

#     if len(last_grad) == 0:
#         return np.nan, grad
#     else:
#         distance = torch.norm(grad - last_grad)
#         return distance, grad


# We use this function to complete the entire
# training process. In order to plot the loss landscape,
# you need to record the loss value of each step.
# Of course, as before, you can test your model
# after drawing a training round and save the curve
# to observe the training
def train(model, optimizer, criterion, train_loader, val_loader, scheduler=None, epochs_n=100, best_model_path=None):
    if type(model) == VGG_A:
        layer_id = 3
    if type(model) == VGG_A_BatchNorm:
        layer_id = 4
    else:
        pass

    model.to(device)
    best_val_acc = 0
    best_val_epoch = 0
    distances = []
    iterations = []
    i = 0
    lrs = [1e-4, 1e-3, 1e-2, 0.1, 0.4]
    train_lr = optimizer.state_dict()['param_groups'][0]['lr']
    print(train_lr)

    for epoch in range(epochs_n):
        if scheduler is not None:
            scheduler.step()
        model.train()

        for data in train_loader:
            x, y = data
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            prediction = model(x)
            loss = criterion(prediction, y)

            loss.backward()
            if i % 30 == 0:
                init_weights = get_vgg_weights(model)
                init_grads = get_vgg_gradients(model)
                init_grad = model.features[layer_id].weight.grad.clone()                

            optimizer.step()

            if i % 30 == 0:
                new_weight = get_vgg_weights(model)
                dis = 0

                for lr in lrs:
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr

                    # update in this lr
                    optimizer.zero_grad()
                    set_vgg_gradients(model, init_grads)
                    set_vgg_weights(model, init_weights)
                    optimizer.step()

                    set_vgg_weights(model, init_weights, feature_border=layer_id)
                    optimizer.zero_grad()
                    prediction = model(x)
                    loss = criterion(prediction, y)
                    loss.backward()
                    new_grad = model.features[layer_id].weight.grad.clone()

                    dis = max(dis, torch.norm(new_grad - init_grad, 2) / lr / torch.norm(init_grad, 2))

                distances.append(dis)
                iterations.append(i)

                set_vgg_weights(model, new_weight)
                for param_group in optimizer.param_groups:
                        param_group['lr'] = train_lr

            i += 1

        val_acc = get_accuracy(val_loader, model)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_val_epoch = epoch

    print("best acc:{0:.3f}\tepoch:{1}".format(best_val_acc, best_val_epoch))

    return iterations, distances


# Use this function to plot the final loss landscape,
# fill the area between the two curves can use plt.fill_between()
def plot_beta(iteration,VGG_A_curve,VGG_A_BN_curve):
    
    plt.style.use('ggplot')
    
   
    plt.plot(iteration, VGG_A_curve, c='green',label='Standard VGG')
    plt.plot(iteration, VGG_A_BN_curve, c='firebrick',label='Standard VGG + BatchNorm')

    plt.xticks(np.arange(0, iteration[-1], 1000))
    plt.xlabel('Steps')
    plt.ylabel('Beta')
    plt.title('Beta smoothness')
    plt.legend(loc='upper right', fontsize='x-large')
    plt.savefig('Beta_Smoothness.png')


# Train your model
# feel free to modify
epo = 20
lrs = [1e-3]
# loss_save_path = ''
# grad_save_path = ''

set_random_seeds(seed_value=2020, device=device)
    
VGG_A_losses = []
VGG_A_BN_losses = []
    
for lr in lrs:
    print("lr:{0}\tVGG_A".format(lr))
    model = VGG_A()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    iterations, VGG_A_loss = train(model, optimizer, criterion, train_loader, val_loader, epochs_n=epo)
    VGG_A_losses.append(VGG_A_loss)
    
    print("lr:{0}\tVGG_A_BatchNorm".format(lr))
    model = VGG_A_BatchNorm()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    _, VGG_A_BN_loss = train(model, optimizer, criterion, train_loader, val_loader, epochs_n=epo)
    VGG_A_BN_losses.append(VGG_A_BN_loss)
   
VGG_A_losses = np.array(VGG_A_losses)
VGG_A_BN_losses = np.array(VGG_A_BN_losses)

VGG_A_max = VGG_A_losses.max(axis=0).astype(float)
VGG_A_BN_max = VGG_A_BN_losses.max(axis=0).astype(float)
    
plot_beta(iterations, VGG_A_max, VGG_A_BN_max)


