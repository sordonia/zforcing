import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.autograd import Variable
import torchvision.datasets as dsets
import time
import click
import numpy
import numpy as np
import os
import random
from itertools import chain
import load
import torch.nn.functional as F
from model import ZForcing
seed = 1234

def evaluate(dataset, model, split='valid'):
    def get_batch():
        if split == 'valid':
            return dataset.get_valid_batch()
        else:
            return dataset.get_test_batch()
    model.eval()
    hidden = model.init_hidden(dataset.batch_size)
    loss = []
    length = 40
    for x, y, x_mask in get_batch():
        l = 0.
        for i in range(0, x.shape[0], length):
            x_ = Variable(torch.from_numpy(x[i:i+length]), volatile=True).float().cuda()
            y_ = Variable(torch.from_numpy(y[i:i+length]), volatile=True).float().cuda()
            x_mask_ = Variable(torch.from_numpy(x_mask[i:i+length]), volatile=True).float().cuda()
            # compute all the states for forward and backward
            fwd_nll, bwd_nll, aux_nll, kld = \
                model(x_, y_, x_mask_, hidden)
            l += (fwd_nll + kld).data[0]
        loss.append(l)
    return np.mean(np.asarray(loss))


@click.command()
@click.option('--model')
@click.option('--data', default='./')
def run(model, data):
    rng = np.random.RandomState(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    model = ZForcing.load(model)
    timit = load.TimitData(data + 'timit_raw_batchsize64_seqlen40.npz', 32)
    model.cuda()
    val_loss = evaluate(timit, model)
    log_line = 'valid -- nll: %f' % val_loss
    print(log_line)
    test_loss = evaluate(timit, model, split='test')
    log_line = 'test -- nll: %f' % test_loss
    print(log_line)

if __name__ == '__main__':
    run()
