from __future__ import print_function, division
import numpy as np
import sys

class metrics(object):
    def __init__(self, maskTh=None):
        self.maskTh = maskTh
        self.iou = {}

    def fastHist(self, a, b, n):
        k = (a >= 0) & (a < n)
        return np.bincount(n * a[k].astype(int) + b[k].astype(int),
            minlength=n**2).reshape(n, n)

    def add(self, gt, pred, score):  # k x n
        pred = pred if self.maskTh is None else (pred>self.maskTh).astype(int)
        iou = np.ones((gt.shape[0], pred.shape[0]), dtype=np.float)
        for i in range(gt.shape[0]):
            for j in range(pred.shape[0]):
                hist = self.fastHist(gt[i], pred[j], 2)
                iou[i,j] = hist[1, 1] / (hist.sum() - hist[0, 0])
        self.iou[score].append(iou)

    def AP(self, iouTh=0.5):
        prec, rec = [], []
        for score in self.iou.keys():
            tp, fp, npos = 0, 0, 0
            for iou in self.iou[score]:
                flag = np.zeros((iou.shape[0],), dtype=int)
                for j in range(iou.shape[1]):
                    if np.max(iou[:,j]) > iouTh and flag[np.argmax(iou[:,j])]:
                        tp += 1
                        flag[np.argmax(iou[:,j])] = 1
                    else:
                        fp += 1
                npos += iou.shape[0]
            rec.append(tp / float(npos))
            prec.append(tp / np.maximum(tp + fp, np.finfo(np.float64).eps))
        return np.nanmean(prec), prec, rec

    def ABO(self, score=None):
        abo = []
        covering = []
        scores = list(self.iou.keys()) if score is None else [score]
        for sc in scores:
            count, countArea, sumIU, sumIUArea = 0, 0, 0, 0
            for iou in self.iou[score]:
                for i in range(iou.shape[0]):
                    sumIU += np.max(iou[i])
                    sumIUArea += np.max(iou[i])*np.sum(iou.shape[0])
                    count += 1
                    countArea += np.sum(iou.shape[0])
            abo.append(sumIU/count)
            covering.append(sumIUArea/countArea)
        return abo, covering


if __name__ == '__main__':
    # gt: [(kg, n), ...]
    # pred: [(kp, n), ...]
    numIm = len(gt)
    for _ in range(numIm):
