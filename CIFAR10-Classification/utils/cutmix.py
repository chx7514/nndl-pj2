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

    return bbx1, bby1, bbx2, bby2

def cutmix_data(x, y, alpha=1.0):
    """
    input: 
        x, y: input sample
        alpha: coefficient of Beta distribution
    output: (mixed_x, y_a, y_b, lambda)
        mixed_x: mixed input
        y_a, y_b: label of two sample
        lamb: coefficient of mixup
    """
    if alpha > 0:
        lamb = np.random.beta(alpha, alpha)
    else:
        lamb = 1

    batch_size = x.size()[0]
    shuf_index = torch.randperm(batch_size).cuda()

    bbx1, bby1, bbx2, bby2 = rand_mask(x.size(), lamb)
    x[:, :, bbx1: bbx2, bby1: bby2] = x[shuf_index, :, bbx1: bbx2, bby1: bby2]

    return x, y, y[shuf_index], lamb

def cutmix_criterion(criterion, logit, y_a, y_b, lamb):
    """
    input: 
        criterion: loss function
        logit: output of model
        y_a, y_b: label of two sample
        lamb: coefficient of cutmix
    output: 
        loss
    """
    return lamb * criterion(logit, y_a) + (1 - lamb) * criterion(logit, y_b)

def cutmix_train(train_loader, model, criterion, optimizer, epoch):
    #每个epoch的优化过程
    losses = AverageMeter()

    # switch to train mode
    model.train()

    for input, target in warp_tqdm(train_loader, True):

        input = input.cuda()
        target = target.cuda()

        # cutmix
        input, target_a, target_b, lamb = cutmix_data(input, target)
        input, target_a, target_b = map(Variable, (input,target_a, target_b))
        
        # compute output
        output = model(input)
        loss = cutmix_criterion(criterion, output, target_a, target_b, lamb)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure accuracy and record loss
        losses.update(loss.item(), input.size(0))

    log = 'Epoch:{0}\tLoss: {loss.avg:.4f}\t'.format(epoch, loss=losses)
    return log