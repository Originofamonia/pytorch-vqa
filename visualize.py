"""
https://tree.rocks/get-heatmap-from-cnn-convolution-neural-network-aka-grad-cam-222e08f57a34
https://www.programmersought.com/article/86424799232/
TODO:
integrate draw_cam.py into here
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
from docx import Document
from docx.shared import Inches

import config
import data2
import model2
import utils

from cam_example import draw_CAM2


def infer(net, loader, tracker, prefix=''):
    """
    infer on train or val splits
    """
    pbar = tqdm(loader, desc=f'len={len(loader)}')
    net.train()
    answ = []
    idxs = []
    counts = []
    # tracker_class, tracker_params = tracker.MeanMonitor, {}
    # loss_tracker = tracker.track('{}_loss'.format(prefix), tracker_class(**tracker_params))
    # acc_tracker = tracker.track('{}_acc'.format(prefix), tracker_class(**tracker_params))

    log_softmax = nn.LogSoftmax(dim=1)
    docu = Document()
    for i, batch in enumerate(pbar):
        if i > 300:
            break
        v, q, a, q_len, idx, img, img_path, q_words, a_words = batch
        img_path = img_path[0]
        q_words = q_words[0] + '?'
        a_words = a_words[0]
        var_params = {
            'requires_grad': False,
        }
        img = Variable(img.cuda(), **var_params)
        q = Variable(q.cuda(), **var_params)
        a = Variable(a.cuda(), **var_params)
        q_len = Variable(q_len.cuda(), **var_params)
        save_path = f'output/heatmap_' + img_path.split('/')[-1].split('.')[0] + '.png'

        out, q_feat, words_feat, a_feat, hv = net(img, q, q_len)
        y_pred = out.argmax(dim=1).detach().cpu().numpy()
        y_pred_words = [loader.dataset.idx_to_answer[item] for item in y_pred]
        q_feat = q_feat.squeeze(0)
        words_feat = words_feat.squeeze(0)
        words_att = []  # cosine similarity
        for j in range(q_len.item()):
            sim = torch.cosine_similarity(q_feat, words_feat[j], dim=0)
            words_att.append(sim.item())
        words_att = np.array(words_att).reshape(1,-1)
        # integrate CAM fn on image here, done
        def extract(g):
            global features_grad
            features_grad = g

        pred = torch.argmax(out).item()
        pred_class = out[:, pred]
        hv.register_hook(extract)  # [1,2048,14,14]
        pred_class.backward()

        grads = features_grad

        pooled_grads = torch.nn.functional.adaptive_avg_pool2d(grads, (1, 1))  # [1,2048,1,1]

        pooled_grads = pooled_grads[0]  # [2048,1,1]
        hv = hv[0]
        # 2048 is the number of channels in the last layer of feature
        for i in range(pooled_grads.size(0)):
            hv[i, ...] *= pooled_grads[i, ...]

        heatmap = hv.detach().cpu().numpy()
        heatmap = np.mean(heatmap, axis=0)
        heatmap = np.maximum(heatmap, 0)
        heatmap /= np.max(heatmap)

        img = plt.imread(img_path, )
        heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0])) # Adjust the size of the heat map to be the same as the original image
        heatmap = np.uint8(255 * heatmap) # Convert the heat map to RGB format
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET) # Apply the heat map to the original image
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        # superimposed_img = heatmap * 0.3 + img # here 0.4 is the heat map intensity factor
        # superimposed_img /= superimposed_img.max()
        # cv2.imwrite(save_path, superimposed_img) # save the image to the hard disk
        fig, axs = plt.subplots(2, 2, figsize=(6, 6)) # (w, h)
        axs[0,0].imshow(img)
        axs[0,0].set_axis_off()
        axs[0,1].imshow(heatmap)
        axs[0,1].set_axis_off()
        axs[1,0].imshow(img)
        axs[1,0].imshow(heatmap, alpha=0.3)
        axs[1,0].set_axis_off()
        axs[1,1].imshow(words_att, cmap='jet', interpolation='nearest')
        axs[1,1].set_yticks(np.arange(1))
        axs[1,1].set_yticklabels(y_pred_words)
        axs[1,1].set_xticks(np.arange(q_len.item()))
        axs[1,1].set_xticklabels(q_words.split(' '),rotation=-40)
        fig.text(0.02,0.91, f'q: {q_words}\ny_true: {a_words}\ny_pred: {y_pred_words[0]}')
        plt.savefig(save_path)
        plt.close(fig)
        docu.add_picture(save_path, width=Inches(6))

    docu.save(f'output/{prefix}_heatmaps.docx')
    counts = np.sum(counts)
    return answ, counts, idxs


def main():
    log = torch.load('D:\CSE6363\hw4\pytorch-vqa\logs/2017-08-04_00.55.19.pth')
    tokens = len(log['vocab']['question']) + 1

    net = torch.nn.DataParallel(model2.Net(tokens))
    # for layer in net.named_parameters():
    #     if layer[0] in log['weights']:
    #         print(layer[0])
    #         layer[1].weight.data.copy_(log['weights'][layer[0]])
    
    net.load_state_dict(log['weights'], strict=False)  # TODO: partial model load weights
    train_loader = data2.get_loader(train=True)
    # val_loader = data2.get_loader(val=True)

    tracker = utils.Tracker()

    # count: correct counts
    ans, counts, idx = infer(net, train_loader, tracker, prefix='train')
    print(f'train set acc: {counts/241918:.3f};')
    # ans, counts, idx = infer(net, val_loader, tracker, prefix='val')
    # print(f'val set acc: {counts/121512:.3f};')


if __name__ == '__main__':
    main()
