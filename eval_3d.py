#!/usr/bin/env python
# -*- coding: utf-8 -*-

import SimpleITK as sitk
import numpy as np
from MLutils.evaluation_metrics3D import Dice, metrics_3d, IoU, over_rate, under_rate
import os
import glob
from scipy.spatial import distance
from numpy.core.umath_tests import inner1d
from skimage import filters

# A = np.array([[1,2],[3,4],[5,6],[7,8]])
# B = np.array([[2,3],[4,5],[6,7],[8,9],[10,11]])

data_path = "/1/code/Seg3D/results_test/NEW/UnetSSoriginal/"



def get_metrics(data_path):
    AUC,ACC, SEN, SPE, IOU, DSC, PRE = [], [], [], [], [] ,[],[]
    for file in glob.glob(os.path.join(data_path, "pred_4400", "*.mha")):
        print(file)
        base_name = os.path.basename(file)[:-15]
        # base_name = os.path.basename(file)
        # label_name = base_name + ".mha"
        label_path = os.path.join(data_path, "label", base_name + '.mha')
        pred = sitk.ReadImage(file)
        label = sitk.ReadImage(label_path)
        # pred_np = np.int64(sitk.GetArrayFromImage(pred))
        pred_np = np.int64(sitk.GetArrayFromImage(pred))
        print(pred_np.max())
        # label_np = np.int64(sitk.GetArrayFromImage(label) * (255 / 1000))
        label_np = np.int64(sitk.GetArrayFromImage(label).transpose(2,1,0))


        # dice = Dice(pred_np, label_np)
        # dice = IoU(pred_np, label_np)
        # hd = HausdorffDist(pred_np, label_np)
        # print(hd)
        acc, sen, spe, iou, dsc, pre = metrics_3d(pred_np, label_np)
        # over_r = over_rate(pred_np, label_np)
        # under_r = under_rate(pred_np, label_np)

        ACC.append(acc)
        SEN.append(sen)
        SPE.append(spe)
        IOU.append(iou)

        DSC.append(dsc)
        PRE.append(pre)
        # AUC.append(auc)

    avgAUC,avgACC, avgSEN, avgSPE, avgIOU, avgDSC, avgPRE = np.mean(AUC),np.mean(ACC), np.mean(SEN), np.mean(SPE), np.mean(IOU), np.mean(DSC), np.mean(
        PRE)
    varAUC,varACC, varSEN, varSPE, varIOU, varDSC, varPRE = np.var(AUC),np.var(ACC), np.var(SEN), np.var(SPE), np.var(IOU), np.var(DSC), np.var(PRE)
    avg = [avgAUC,avgACC, avgSEN, avgSPE, avgIOU, avgDSC, avgPRE]
    var = [varAUC,varACC, varSEN, varSPE, varIOU, varDSC, varPRE]

    return avg, var


avg, var = get_metrics(data_path)

avgAUC,avgACC, avgSEN, avgSPE, avgIOU, avgDSC, avgPRE= avg
varAUC,varACC, varSEN, varSPE, varIOU, varDSC, varPRE = var
print("AuC = {0:.4f}% \u00b1 {1:.4f}".format(avgAUC * 100, varAUC * 100))

print("ACC = {0:.4f}% \u00b1 {1:.4f}".format(avgACC * 100, varACC * 100))
print("SEN = {0:.4f}% \u00b1 {1:.4f}".format(avgSEN * 100, varSEN * 100))
print("SPE = {0:.4f}% \u00b1 {1:.4f}".format(avgSPE * 100, varSPE * 100))
print("IOU = {0:.4f}% \u00b1 {1:.4f}".format(avgIOU * 100, varIOU * 100))
print("DSC = {0:.4f}% \u00b1 {1:.4f}".format(avgDSC * 100, varDSC * 100))
print("PRE = {0:.4f}% \u00b1 {1:.4f}".format(avgPRE * 100, varPRE * 100))
