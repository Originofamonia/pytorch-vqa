"""
https://tree.rocks/get-heatmap-from-cnn-convolution-neural-network-aka-grad-cam-222e08f57a34
https://www.programmersought.com/article/86424799232/
TODO:
1. add resnet as the feature extractor
2. write new Dataset class to directly read image files
"""
import cv2
import matplotlib.pyplot as plt
from scipy.ndimage import zoom

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
import data2
import model2
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
        v, q, a, q_len, idx, img, img_path, q_words, a_words = batch
        var_params = {
            'requires_grad': False,
        }
        img = Variable(img.cuda(), **var_params)
        q = Variable(q.cuda(), **var_params)
        a = Variable(a.cuda(), **var_params)
        q_len = Variable(q_len.cuda(), **var_params)

        out, q_feat, words_feat, a_feat, v2 = net(img, q, q_len)
        y_pred = out.argmax(dim=1).detach().cpu().numpy()
        y_pred_words = [loader.dataset.idx_to_answer[item] for item in y_pred]


def main():
    log = torch.load('D:\CSE6363\hw4\pytorch-vqa\logs/2017-08-04_00.55.19.pth')
    tokens = len(log['vocab']['question']) + 1

    net = torch.nn.DataParallel(model2.Net(tokens))
    net.load_state_dict(log['weights'])  # TODO: partial model load weights

    # train_loader = data.get_loader(train=True)
    val_loader = data2.get_loader(val=True)

    tracker = utils.Tracker()

    # count: correct counts
    # ans, counts, idx = infer(net, train_loader, tracker, prefix='train')
    # print(f'train set acc: {counts/241918:.3f};')
    ans, counts, idx = infer(net, val_loader, tracker, prefix='val')
    print(f'val set acc: {counts/121512:.3f};')


if __name__ == '__main__':
    main()
