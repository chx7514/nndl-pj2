import torch
import numpy as np
from torch.autograd import Variable
from utils.util import AverageMeter, warp_tqdm

def rand_mask(size, lamb):
    """
    input:
        size: size of the samples
        lamb: remaining area
    output:
        mask: cutout mask
    """
    W, H = size[2], size[3]
    cut = np.sqrt(1. - lamb)
    cut_w = np.int(W * cut)
    cut_h = np.int(H * cut)

    # center position
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    mask = np.ones(size, np.float32)
    mask[:, :, bbx1: bbx2, bby1: bby2] = 0.
    mask = torch.from_numpy(mask)
    return mask

def cutout_data(x, alpha=1.0):
    '''Returns cutout inputs'''
    if alpha > 0:
        lamb = np.random.beta(alpha, alpha)
    else:
        lamb = 1

    mask = rand_mask(x.size(), lamb)

    mask = mask.cuda()
    x_cutout = x * mask
       
    return x_cutout

def cutout_train(train_loader, model, criterion, optimizer, epoch):
    #每个epoch的优化过程
    losses = AverageMeter()

    # switch to train mode
    model.train()

    for input, target in warp_tqdm(train_loader, True):

        input = input.cuda()
        target = target.cuda()

        # cutout
        input = cutout_data(input)
        input, target = map(Variable, (input,target))
        
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
    return log