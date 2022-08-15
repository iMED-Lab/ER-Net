#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import SimpleITK as sitk
import glob
import os
from scipy.spatial import distance
from sklearn.metrics import f1_score
from sklearn import metrics
# from skimage import filters


def AUC_score(SR, GT, threshold=0.5):
    # SR = SR.numpy()
    GT = GT.ravel()  # we want to make them into vectors
    SR = SR.ravel()  # .detach()
    # fpr, tpr, _ = metrics.roc_curve(GT, SR)
    # fpr, tpr, _ = metrics.roc_curve(SR, GT)
    # roc_auc = metrics.auc(fpr, tpr)
    roc_auc = metrics.roc_auc_score(GT, SR)
    return roc_auc


def numeric_score(pred, gt):
    FP = np.float(np.sum((pred == 255) & (gt == 0)))
    FN = np.float(np.sum((pred == 0) & (gt == 255)))
    TP = np.float(np.sum((pred == 255) & (gt == 255)))
    TN = np.float(np.sum((pred == 0) & (gt == 0)))
    return FP, FN, TP, TN


def Dice(pred, gt):
    pred = np.int64(pred / 255)
    gt = np.int64(gt / 255)
    dice = np.sum(pred[gt == 1]) * 2.0 / (np.sum(pred) + np.sum(gt))
    return dice


def IoU(pred, gt):
    pred = np.int64(pred / 255)
    gt = np.int64(gt / 255)
    m1 = np.sum(pred[gt == 1])
    m2 = np.sum(pred == 1) + np.sum(gt == 1) - m1
    iou = m1 / m2
    return iou


def metrics_3d(pred, gt):
    # auc = AUC_score(pred/255, gt)
    # pred = (pred.data.cpu().numpy() * 255).astype(np.uint8)
    # # input("wait2..")
    # gt = (gt.data.cpu().numpy() * 255).astype(np.uint8)
    # threshold = filters.threshold_otsu(pred, nbins=256)
    # print('threshold',threshold)
    # if pred.min() == pred.max():
    pred = np.where(pred > 150, 255.0, 0)  ########
    # print(pred.shape)
    # else:
    #     threshold = filters.threshold_otsu(pred, nbins=256)
    #     pred = np.where(pred > threshold, 255.0, 0)  ########


    FP, FN, TP, TN = numeric_score(pred, gt)
    #acc = (TP + TN) / (TP + FP + FN + TN + 1e-10)
    sen = TP / (TP + FN + 1e-10)  # recall sensitivity
    spe = TN / (TN + FP + 1e-10)
    #iou = TP / (TP + FN + FP + 1e-10)
    dsc = 2.0 * TP / (TP * 2.0 + FN + FP + 1e-10)
    #pre = TP / (TP + FP + 1e-10)

    return sen, spe, dsc


def over_rate(pred, gt):
    # pred = np.int64(pred / 255)
    # gt = np.int64(gt / 255)
    Rs = np.float(np.sum(gt == 255))
    Os = np.float(np.sum((pred == 255) & (gt == 0)))
    OR = Os / (Rs + Os)
    return OR


def under_rate(pred, gt):
    # pred = np.int64(pred / 255)
    # gt = np.int64(gt / 255)
    Rs = np.float(np.sum(gt == 255))
    Us = np.float(np.sum((pred == 0) & (gt == 255)))
    Os = np.float(np.sum((pred == 255) & (gt == 0)))
    UR = Us / (Rs + Os)
    return UR
