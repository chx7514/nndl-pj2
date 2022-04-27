import torch
import numpy as np
from torch.autograd import Variable
from utils.util import AverageMeter, warp_tqdm

def mixup_data(x, y, alpha=1.0):
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

    mixed_x = lamb * x + (1 - lamb) * x[shuf_index, :]
    return mixed_x, y, y[shuf_index], lamb

def mixup_criterion(criterion, logit, y_a, y_b, lamb):
    """
    input: 
        criterion: loss function
        logit: output of model
        y_a, y_b: label of two sample
        lamb: coefficient of mixup
    output: 
        loss
    """
    return lamb * criterion(logit, y_a) + (1 - lamb) * criterion(logit, y_b)


def mixup_train(train_loader, model, criterion, optimizer, epoch):
    #每个epoch的优化过程
    losses = AverageMeter()

    # switch to train mode
    model.train()

    for input, target in warp_tqdm(train_loader, True):

        input = input.cuda()
        target = target.cuda()

        # mixup
        input, target_a, target_b, lamb = mixup_data(input, target)
        input, target_a, target_b = map(Variable, (input,target_a, target_b))
        
        # compute output
        output = model(input)
        loss = mixup_criterion(criterion, output, target_a, target_b, lamb)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure accuracy and record loss
        losses.update(loss.item(), input.size(0))

    log = 'Epoch:{0}\tLoss: {loss.avg:.4f}\t'.format(epoch, loss=losses)
    return log