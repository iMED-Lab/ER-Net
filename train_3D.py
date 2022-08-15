#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author : Hao Zhang
# @File   : train3d.py
"""
Training script for ER-Net
"""
import os
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
import visdom
import numpy as np
from vis import Visualizeer
from models.ER_Net import ER_Net
from DataLoader.NerveLoader import Data
# from DataLoader.VesselLoader import Data
from util.train_metrics import metrics, metrics3d
from util.losses import dice_coeff_loss, boundary_loss_func, boundary_loss_Dice  ##############
from util.visualize import init_visdom_line, update_lines
import csv

args = {
    'root': '/home/PycharmProjects/Attention/',
    'data_path': './data/',
    'fold': 3,
    'epochs': 700,
    'lr': 0.0001,
    'snapshot': 50,
    'test_step': 2,
    'ckpt_path': './checkpoint/',
    'batch_size': 1,
    'env': 'ER_NET'
}
# Setup CUDA device(s)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def save_ckpt(net, iter):
    if not os.path.exists(args['ckpt_path']):
        os.makedirs(args['ckpt_path'])
    torch.save(net, args['ckpt_path'] + '/Net' + iter + '.pkl')
    print("{} Saved model to:{}".format("\u2714", args['ckpt_path']))


# adjust learning rate (poly)
def adjust_lr(optimizer, base_lr, iter, max_iter, power=0.9):
    lr = base_lr * (1 - float(iter) / max_iter) ** power
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def train():
    net = ER_Net(classes= 1,channels=1).cuda()
    net = nn.DataParallel(net).cuda()
    optimizer = optim.Adam(net.parameters(), lr=args['lr'], weight_decay=0.0005)
    critrion = boundary_loss_func().cuda()
    iters = 1
    best_sen, best_dsc = 0., 0.

    train_data = Data(args['data_path'], train=True, fold=args['fold'])
    batchs_data = DataLoader(train_data, batch_size=args['batch_size'], num_workers=4, shuffle=False)
    for epoch in range(args['epochs']):
        net.train()
        for idx, batch in enumerate(batchs_data):
            image = batch[0].type(torch.FloatTensor).cuda()
            label = batch[1].type(torch.FloatTensor).cuda()
            optimizer.zero_grad()
            pred = net(image)
            # loss = F.cross_entropy(pred, label).cuda()
            loss1 = dice_coeff_loss(pred, label).cuda()
            sen, spe, dsc = metrics3d(pred, label, pred.shape[0])
            if dsc < 0.8:  ##  it can be 0.6, 0.7, 0.8;adjust to your data set
            #     # loss = 0.8 * loss1 + 0.2 * loss3
                loss = loss1

            else:
                loss2 = critrion(pred, label)
                loss = loss1 + loss2
            # loss = critrion(pred, label)
            loss.backward()
            optimizer.step()
            print('{0:d}:{1:d}] \u2501\u2501\u2501 loss:{2:.10f}\tsen:{3:.4f}\tspe:{4:.4f}\tdsc:{5:.4f}'.format
                  (epoch + 1, iters, loss.item(), sen / pred.shape[0], spe / pred.shape[0], dsc / pred.shape[0]))
            iters += 1

        adjust_lr(optimizer, base_lr=args['lr'], iter=epoch, max_iter=args['epochs'], power=0.9)
        if (epoch + 1) % args['snapshot'] == 0:
            save_ckpt(net, str(epoch + 1))
        # model eval
        if (epoch + 1) % args['test_step'] == 0:
            test_sen, test_spe, test_dsc = model_eval(net, iters)

            if test_dsc > best_dsc:
                save_ckpt(net, "best_DSC")
                best_dsc = test_dsc
            print(
                "Average SEN:{0:.4f}, average SPE:{1:.4f},average DSC:{2:.4f}".format(
                    test_sen, test_spe, test_dsc))


def model_eval(net, iters):
    test_data = Data(args['data_path'], train=False, fold=args['fold'])
    batchs_data = DataLoader(test_data, batch_size=1)
    net.eval()
    AUC, ACC, SEN, SPE, IOU, DSC, PRE = [], [], [], [], [], [], []
    file_num = 0
    with torch.no_grad():
        for idx, batch in enumerate(batchs_data):
            image = batch[0].float().cuda()
            label = batch[1].cuda().float()
            pred = net(image)
            sen, spe, dsc = metrics3d(pred, label, pred.shape[0])
            SEN.append(sen)
            SPE.append(spe)
            DSC.append(dsc)
            file_num += 1
        return np.mean(SEN), np.mean(SPE), np.mean(DSC)


if __name__ == '__main__':
    train()
