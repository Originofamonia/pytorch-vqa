"""
load pretrained model from https://github.com/Cyanogenoid/pytorch-vqa/releases
"""

import sys
import os.path
import math
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
from tqdm import tqdm

import config
import data
import model
import utils


def infer(net, loader, tracker, prefix=''):
    """
    infer on train or val splits
    """
    pbar = tqdm(loader, desc=f'len={len(loader)}')
    net.eval()
    answ = []
    idxs = []
    counts = []
    tracker_class, tracker_params = tracker.MeanMonitor, {}
    loss_tracker = tracker.track('{}_loss'.format(prefix), tracker_class(**tracker_params))
    acc_tracker = tracker.track('{}_acc'.format(prefix), tracker_class(**tracker_params))

    log_softmax = nn.LogSoftmax(dim=1)

    for batch in pbar:
        v, q, a, q_len, idx, image_id, q_words, a_words = batch
        var_params = {
            'requires_grad': False,
        }
        v = Variable(v.cuda(), **var_params)
        q = Variable(q.cuda(), **var_params)
        a = Variable(a.cuda(), **var_params)
        q_len = Variable(q_len.cuda(), **var_params)

        out, q_feat, words_feat, a_feat = net(v, q, q_len)
        y_pred = out.argmax(dim=1).detach().cpu().numpy()
        y_pred_words = [loader.dataset.idx_to_answer[item] for item in y_pred]
        nll = -log_softmax(out)
        loss = (nll * a / 10).sum(dim=1).mean()
        acc = utils.batch_accuracy(out.data, a.data).cpu()

        _, answer = out.data.cpu().max(dim=1)
        # answ.append(answer.view(-1))
        counts.append(acc.view(-1).sum().item())
        # idxs.append(idx.view(-1).clone())

        loss_tracker.append(loss.item())
        # acc_tracker.append(acc.mean())
        for a in acc:
            acc_tracker.append(a.item())
        fmt = '{:.3f}'.format
        pbar.set_postfix(loss=fmt(loss_tracker.mean.value), acc=fmt(acc_tracker.mean.value))

    # answ = list(torch.cat(answ, dim=0))
    counts = np.sum(counts)
    # idxs = list(torch.cat(idxs, dim=0))
    return answ, counts, idxs


def main():
    log = torch.load('D:\CSE6363\hw4\pytorch-vqa\logs/2017-08-04_00.55.19.pth')
    tokens = len(log['vocab']['question']) + 1

    net = torch.nn.DataParallel(model.Net(tokens))
    net.load_state_dict(log['weights'])

    train_loader = data.get_loader(train=True)
    val_loader = data.get_loader(val=True)

    tracker = utils.Tracker()

    # count: correct counts
    ans, counts, idx = infer(net, train_loader, tracker, prefix='train')
    print(f'train set acc: {counts/241918:.3f};')
    ans, counts, idx = infer(net, val_loader, tracker, prefix='val')
    print(f'val set acc: {counts/121512:.3f};')
    

if __name__ == '__main__':
    main()
