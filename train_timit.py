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


def evaluate(dataset, model, split='valid'):
    def get_batch():
        if split == 'valid':
            return dataset.get_valid_batch()
        else:
            return dataset.get_test_batch()

    model.eval()
    hidden = model.init_hidden(dataset.batch_size)
    loss = []
    for x, y, x_mask in get_batch():
        x = Variable(torch.from_numpy(x), volatile=True).float().cuda()
        y = Variable(torch.from_numpy(y), volatile=True).float().cuda()
        x_mask = Variable(torch.from_numpy(x_mask), volatile=True).float().cuda()

        # compute all the states for forward and backward
        fwd_nll, bwd_nll, aux_nll, kld = \
            model(x, y, x_mask, hidden)
        loss.append((fwd_nll + kld).data[0])
    return np.mean(np.asarray(loss))


@click.command()
@click.option('--expname', default='timit_logs')
@click.option('--nlayers', default=1)
@click.option('--seed', default=1234)
@click.option('--num_epochs', default=100)
@click.option('--rnn_dim', default=1024)
@click.option('--data', default='./')
@click.option('--bsz', default=32)
@click.option('--lr', default=0.001)
@click.option('--z_dim', default=256)
@click.option('--emb_dim', default=512)
@click.option('--mlp_dim', default=512)
@click.option('--aux_end', default=0.0)
@click.option('--aux_sta', default=0.0)
@click.option('--kla_sta', default=0.2)
@click.option('--bwd', default=0.)
@click.option('--cond_ln', is_flag=True)
@click.option('--z_force', is_flag=True)
def train(expname, nlayers, seed, num_epochs, rnn_dim, data, bsz, lr, z_dim,
          emb_dim, mlp_dim, aux_end, aux_sta, kla_sta, bwd, cond_ln, z_force):
    rng = np.random.RandomState(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    log_interval = 10
    model_id = 'timit_seed{}_cln{}_zf{}_auxsta{}_auxend{}_klasta{}_bwd{}'.format(
            seed, int(cond_ln), z_force, aux_sta, aux_end, kla_sta, bwd)
    if not os.path.exists(expname):
        os.makedirs(expname)
    log_file_name = os.path.join(expname, model_id + '.txt')
    model_file_name = os.path.join(expname, model_id + '.pt')
    log_file = open(log_file_name, 'w')

    model = ZForcing(200, emb_dim, rnn_dim, z_dim,
                     mlp_dim, 400, nlayers=nlayers,
                     cond_ln=cond_ln)
    model.z_force = z_force
    print('Loading data..')
    timit = load.TimitData(data + 'timit_raw_batchsize64_seqlen40.npz', bsz)
    print('Done.')
    model.cuda()
    hidden = model.init_hidden(bsz)
    opt = torch.optim.Adam(model.parameters(), lr=lr, eps=1e-5)

    kld_step = 0.00005
    aux_step = 0.00005
    kld_weight = kla_sta
    aux_weight = aux_sta
    nbatches = timit.u_train.shape[0] // bsz
    t = time.time()
    for epoch in range(num_epochs):
        step = 0
        old_valid_loss = np.inf
        b_fwd_loss, b_bwd_loss, b_kld_loss, b_aux_loss, b_all_loss = \
            (0., 0., 0., 0., 0.)
        model.train()
        print('Epoch {}: ({})'.format(epoch, model_id.upper()))
        for x, y, x_mask in timit.get_train_batch():
            step += 1
            opt.zero_grad()
            x = Variable(torch.from_numpy(x)).float().cuda()
            y = Variable(torch.from_numpy(y)).float().cuda()
            x_mask = Variable(torch.from_numpy(x_mask)).float().cuda()

            # compute all the states for forward and backward
            fwd_nll, bwd_nll, aux_nll, kld = model(x, y, x_mask, hidden)
            bwd_nll = (aux_weight > 0.) * (bwd * bwd_nll)
            aux_nll = aux_weight * aux_nll
            all_loss = fwd_nll + bwd_nll + aux_nll + kld_weight * kld
            # anneal kld cost
            kld_weight += kld_step
            kld_weight = min(kld_weight, 1.)
            # anneal auxiliary cost
            if aux_sta <= aux_end:
                aux_weight += aux_step
                aux_weight = min(aux_weight, aux_end)
            else:
                aux_weight -= aux_step
                aux_weight = max(aux_weight, aux_end)

            if np.isnan(all_loss.data[0]) or np.isinf(all_loss.data[0]):
                continue

            all_loss.backward()
            torch.nn.utils.clip_grad_norm(model.parameters(), 100.)
            opt.step()

            b_all_loss += all_loss.data[0]
            b_fwd_loss += fwd_nll.data[0]
            b_bwd_loss += bwd_nll.data[0]
            b_kld_loss += kld.data[0]
            b_aux_loss += aux_nll.data[0]

            if step % log_interval == 0:
                s = time.time()
                log_line = 'epoch: [%d/%d], step: [%d/%d], loss: %f, fwd loss: %f, aux loss: %f, bwd loss: %f, kld: %f, kld weight: %f, aux weight: %.4f, %.2fit/s' % (
                    epoch, num_epochs, step, nbatches,
                    b_all_loss / log_interval,
                    b_fwd_loss / log_interval,
                    b_aux_loss / log_interval,
                    b_bwd_loss / log_interval,
                    b_kld_loss / log_interval,
                    kld_weight,
                    aux_weight,
                    log_interval / (s - t))
                b_all_loss = 0.
                b_fwd_loss = 0.
                b_bwd_loss = 0.
                b_aux_loss = 0.
                b_kld_loss = 0.
                t = time.time()
                print(log_line)
                log_file.write(log_line + '\n')
                log_file.flush()

        # evaluate per epoch
        print('--- Epoch finished ----')
        val_loss = evaluate(timit, model)
        log_line = 'valid -- epoch: %s, nll: %f' % (epoch, val_loss)
        print(log_line)
        log_file.write(log_line + '\n')
        test_loss = evaluate(timit, model, split='test')
        log_line = 'test -- epoch: %s, nll: %f' % (epoch, test_loss)
        print(log_line)
        log_file.write(log_line + '\n')
        log_file.flush()

        if old_valid_loss > val_loss:
            old_valid_loss = val_loss
            model.save(model_file_name)
        else:
            for param_group in opt.param_groups:
                lr = param_group['lr']
                if lr > 0.0001:
                    lr *= 0.5
                param_group['lr'] = lr


if __name__ == '__main__':
    train()
