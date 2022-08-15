import numpy as np
import os
import torch.nn as nn
import torch
from PIL import ImageOps, Image
from sklearn.metrics import confusion_matrix

from MLutils.evaluation_metrics3D import metrics_3d, Dice

from sklearn import metrics
# from skimage import filters


def AUC_score(SR, GT, threshold=0.5):
    # SR = SR.numpy()
    GT = GT.ravel()  # we want to make them into vectors
    SR = SR.ravel()  # .detach()
    # fpr, tpr, _ = metrics.roc_curve(GT, SR)
    # fpr, tpr, _ = metrics.roc_curve(SR, GT)
    # roc_auc = metrics.auc(fpr, tpr)
    # roc_auc = metrics()
    roc_auc = metrics.roc_auc_score(GT, SR)
    return roc_auc


def threshold(image):
    image[image >= 100] = 255
    image[image < 100] = 0
    return image


def numeric_score(pred, gt):
    FP = np.float(np.sum((pred == 255) & (gt == 0)))
    FN = np.float(np.sum((pred == 0) & (gt == 255)))
    TP = np.float(np.sum((pred == 255) & (gt == 255)))
    TN = np.float(np.sum((pred == 0) & (gt == 0)))
    return FP, FN, TP, TN


def metrics1(pred, label, batch_size):
    # pred = torch.argmax(pred, dim=1) # for CE Loss series
    outputs = (pred.data.cpu().numpy() * 255).astype(np.uint8)
    labels = (label.data.cpu().numpy() * 255).astype(np.uint8)
    outputs = outputs.squeeze(1)  # for MSELoss()
    labels = labels.squeeze(1)  # for MSELoss()
    outputs = threshold(outputs)  # for MSELoss()

    Acc, SEn = 0., 0.
    for i in range(batch_size):
        img = outputs[i, :, :]
        gt = labels[i, :, :]
        acc, sen = get_acc(img, gt)
        Acc += acc
        SEn += sen
    return Acc, SEn


def metrics3dmse(pred, label, batch_size):
    outputs = (pred.data.cpu().numpy() * 255).astype(np.uint8)
    labels = (label.data.cpu().numpy() * 255).astype(np.uint8)
    outputs = outputs.squeeze(1)  # for MSELoss()
    labels = labels.squeeze(1)  # for MSELoss()
    outputs = threshold(outputs)  # for MSELoss()

    tp, fn, fp, IoU = 0, 0, 0, 0
    for i in range(batch_size):
        img = outputs[i, :, :, :]
        gt = labels[i, :, :, :]
        tpr, fnr, fpr, iou = metrics_3d(img, gt)
        # dcr = Dice(img, gt)
        tp += tpr
        fn += fnr
        fp += fpr
        IoU += iou
    return tp, fn, fp, IoU


def metrics3d(pred, label, batch_size):
    # pred = torch.argmax(pred, dim=1)  # for CE loss series
    # print((pred.data.cpu().numpy() * 255).astype(np.uint8))
    # print('1',pred.max(), pred.min())
    # print(pred.shape)

    pred = (pred.data.cpu().numpy() * 255).astype(np.uint8)
    # print(pred.max(), pred.min())
    # input("wait2..")
    # print(np.unique(label))
    label = (label.data.cpu().numpy() * 255).astype(np.uint8)  # for miccai data
    # label = (label.data.cpu().numpy()).astype(np.uint8)  # for alex  data

    # label = label* 255
    # print('label:', label.max(), np.unique(label))
    # auc = AUC_score(pred, label)

    # outputs = outputs.squeeze(1)  # for MSELoss()
    # labels = labels.squeeze(1)  # for MSELoss()
    # outputs = threshold(outputs)  # for MSELoss()

    auc, acc, sen, spe, iou, dsc, pre = 0, 0, 0, 0, 0, 0, 0
    for i in range(batch_size):
        img = pred[i, :, :, :]
        gt = label[i, :, :, :]
        # AUC = AUC_score(img/255, gt/255)
        SEN, SPE, DSC = metrics_3d(img, gt)
        # dcr = Dice(img, gt)
        # auc += AUC
        # acc += ACC
        sen += SEN
        spe += SPE
        # iou += IOU
        dsc += DSC
        # pre += PRE
    return sen, spe, dsc


def get_acc(image, label):
    image = threshold(image)
    FP, FN, TP, TN = numeric_score(image, label)
    acc = (TP + TN) / (TP + FN + TN + FP + 1e-10)
    sen = (TP) / (TP + FN + 1e-10)
    return acc, sen
