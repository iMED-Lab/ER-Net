import numpy as np
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
import os
import glob
import cv2

path = '/1/code/Seg3D/results_test/'


# path = '../assets/STARE/'
# path = '../assets/NERVE/'
# path = '/home/imed/Documents/Experiments/RRM-Net/assets/DRIVE/'


# path = '../assets/SVP_All/'
# path = '/home/imed/Desktop/unet/stare/'


def eval(path):
    pred_path = path + 'pred/'
    label_path = path + 'label/'
    Acc, Sen, Spe, F1 = 0, 0, 0, 0
    file_num = 0
    for file in glob.glob(os.path.join(pred_path, '*otsu.png')):
        print(file)
        index = os.path.basename(file)[:-14]
        # index = os.path.basename(file)[:-9]
        label_name = os.path.join(label_path, index + '.png')
        pred_img = cv2.imread(file)
        label_img = cv2.imread(label_name)
        pred = pred_img.flatten()
        label = label_img.flatten()

        f1 = metrics.f1_score(label, pred, average='weighted')
        F1 += f1

        tn, fp, fn, tp = metrics.confusion_matrix(label, pred).ravel()
        acc = (tp + tn) / (tp + fn + tn + fp)
        sen = tp / (tp + fn)
        sp = tn / (tn + fp)
        Acc += acc
        Sen += sen
        Spe += sp
        file_num += 1
    print('Acc: {0:.4f}'.format(Acc / file_num))
    print('Sen: {0:.4f}'.format(Sen / file_num))
    print('Sp: {0:.4f}'.format(Spe / file_num))
    print('F1: {0:.4f}'.format(F1 / file_num))


def roc_auc(path, dir):
    pred_path = path + dir
    label_path = path + 'label/'
    FPR, TPR = [], []
    AUC = 0.
    AUC_s = 0.
    file_num = 0
    for file in glob.glob(os.path.join(pred_path, '*pred.png')):
        index = os.path.basename(file)[:-9]
        label_name = os.path.join(label_path, index + '.png')
        pred_img = cv2.imread(file)
        label_img = cv2.imread(label_name)
        pred = pred_img.flatten()
        label = label_img.flatten()
        fpr, tpr, thresholds = metrics.roc_curve(label / 255, pred / 255, drop_intermediate=False)
        auc = metrics.auc(fpr, tpr)
        AUC_s += auc
        # if auc > AUC:
        #     AUC = auc
        #     FPR = fpr
        #     TPR = tpr
        file_num += 1
    avg_auc = AUC_s / file_num
    return FPR, TPR, avg_auc


def roc_auc2(path, dir):
    pred_path = path + dir
    label_path = path + 'label/'
    FPR, TPR = [], []
    AUC = 0.
    file_num = 0
    for file in glob.glob(os.path.join(pred_path, '*otsu.png')):
        index = os.path.basename(file)[:-14]
        label_name = os.path.join(label_path, index + '.png')
        pred_img = cv2.imread(file)
        label_img = cv2.imread(label_name)
        pred = pred_img.flatten()
        label = label_img.flatten()
        fpr, tpr, thresholds = metrics.roc_curve(label / 255, pred / 255, drop_intermediate=False)
        auc = metrics.auc(fpr, tpr)
        if auc > AUC:
            AUC = auc
            FPR = fpr
            TPR = tpr
        file_num += 1
    return FPR, TPR, AUC


def merge_roc():
    path = '/1/code/Seg3D/results_test/pred/'
    path1 = path + 'DRIVE(ddnet_randomwalk)/'
    path2 = path + 'STARE(ddnet_randomwalk)/'
    path3 = path + 'CHASEDB1(ddnet_randomwalk)/'
    path4 = path + 'DRIVE(Unet)/'
    path5 = path + 'STARE/'
    path6 = path + 'CHASEDB1/'

    fpr1, tpr1, auc1 = roc_auc(path1, dir='pred/')
    # fpr11, tpr11, auc11 = roc_auc2(path1, dir='random_walk/')
    fpr2, tpr2, auc2 = roc_auc(path2, dir='pred/')
    # fpr22, tpr22, auc22 = roc_auc2(path2, dir='random_walk/')
    fpr3, tpr3, auc3 = roc_auc(path3, dir='pred/')
    # fpr33, tpr33, auc33 = roc_auc2(path3, dir='random_walk/')
    fpr4, tpr4, auc4 = roc_auc(path4, dir='pred/')
    # fpr44, tpr44, auc44 = roc_auc2(path4, dir='random_walk/')
    fpr5, tpr5, auc5 = roc_auc(path5, dir='pred/')
    # fpr55, tpr55, auc55 = roc_auc2(path5, dir='random_walk/')
    fpr6, tpr6, auc6 = roc_auc(path6, dir='pred/')
    # fpr66, tpr66, auc66 = roc_auc2(path6, dir='random-walk/')

    plt.figure()
    plt.plot(fpr4, tpr4, label='UNet-DRIVE(auc={0:.4f})'.format(auc4), color='red', linestyle='--', linewidth=1)
    # plt.plot(fpr44, tpr44, label='UNet(RW)-DRIVE(auc={0:.4f})'.format(auc44), color='red', linestyle='-.', linewidth=1)
    plt.plot(fpr1, tpr1, label='DDNet-DRIVE(auc={0:.4f})'.format(0.9869), color='red', linestyle='-', linewidth=1)
    # plt.plot(fpr11, tpr11, label='DDNet(RW)-DRIVE(auc={0:.4f})'.format(auc11), color='red', linestyle='-.', linewidth=1)
    plt.plot(fpr5, tpr5, label='UNet-STARE(auc={0:.4f})'.format(auc5), color='lime', linestyle='--', linewidth=1)
    # plt.plot(fpr55, tpr55, label='UNet(RW)-STARE(auc={0:.4f})'.format(auc55), color='lime', linestyle='-.', linewidth=1)
    plt.plot(fpr2, tpr2, label='DDNet-STARE(auc={0:.4f})'.format(auc2), color='lime', linestyle='-', linewidth=1)
    # plt.plot(fpr22, tpr22, label='DDNet(RW)-STARE(auc={0:.4f})'.format(auc22), color='lime', linestyle='-.', linewidth=1)
    plt.plot(fpr6, tpr6, label='UNet-CHASEDB1(auc={0:.4f})'.format(auc6), color='deepskyblue', linestyle='--',
             linewidth=1)
    # plt.plot(fpr66, tpr66, label='UNet(RW)-CHASEDB1(auc={0:.4f})'.format(auc66), color='deepskyblue', linestyle='-.', linewidth=1)
    plt.plot(fpr3, tpr3, label='DDNet-CHASEDB1(auc={0:.4f})'.format(auc3), color='deepskyblue', linestyle='-',
             linewidth=1)
    # plt.plot(fpr33, tpr33, label='DDNet(RW)-CHASEDB1(auc={0:.4f})'.format(auc33), color='deepskyblue', linestyle='-.', linewidth=1)
    plt.xlim(-0.002, 1.)
    plt.ylim(0., 1.005)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right')
    plt.savefig('/home/leila/Desktop/roc_auc.png', dpi=1200)


if __name__ == '__main__':
    eval(path)
    fpr, tpr, auc = roc_auc(path, 'pred/')
    print(auc)
    # merge_roc()
