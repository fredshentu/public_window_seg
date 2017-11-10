from __future__ import print_function, division
import numpy as np

class Metrics(object):
    def __init__(self, maskTh=None):
        self.maskTh = maskTh
        self.iou = {}

    def binaryIoU(self, m1, m2):
        if np.sum(np.logical_or(m1, m2)) == 0:
            return 1.0
        return np.sum(np.logical_and(m1, m2)) / np.sum(np.logical_or(m1, m2))

    def add(self, gt, pred, score):
        pred = pred if self.maskTh is None else (pred>self.maskTh).astype(int)
        iou = np.ones((gt.shape[0], pred.shape[0]), dtype=np.float)
        for i in range(gt.shape[0]):
            for j in range(pred.shape[0]):
                iou[i,j] = self.binaryIoU(gt[i], pred[j])
        if score not in self.iou:
            self.iou[score] = []
        self.iou[score].append(iou)

    def AP(self, iouTh=0.5):
        prec, rec = [], []
        for score in sorted(list(self.iou.keys())):
            tp, fp, npos = 0, 0, 0
            for iou in self.iou[score]:
                flag = np.zeros((iou.shape[0],), dtype=int)
                for j in range(iou.shape[1]):
                    if np.max(iou[:,j])>iouTh and not flag[np.argmax(iou[:,j])]:
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
        scores = sorted(list(self.iou.keys())) if score is None else [score]
        for sc in scores:
            count, countArea, sumIU, sumIUArea = 0, 0, 0, 0
            for iou in self.iou[sc]:
                for i in range(iou.shape[0]):
                    sumIU += np.max(iou[i])
                    sumIUArea += np.max(iou[i])*np.sum(iou.shape[0])
                    count += 1
                    countArea += np.sum(iou.shape[0])
            abo.append(sumIU/count)
            covering.append(sumIUArea/countArea)
        return abo, covering


if __name__ == '__main__':
    import argparse
    import time
    startTime = time.time()
    parser = argparse.ArgumentParser(description="Metric Evaluator")
    parser.add_argument('-gp', '--gtpath', type=str, help="gt path",
        default="./evaluate_models/ground_truth_msk.npy")
    parser.add_argument('-mp', '--mpath', type=str, help="model path")
    parser.add_argument('-k', '--maxprop', type=int, default=20,
        help="max limit on number of predicted proposals per image")
    args = parser.parse_args()

    print('loading models ...')
    gtList = np.load(args.gtpath)  # gt: N x k x h x w  -->  k x n
    modelList = np.load(args.mpath)  # model: N x th x k x h x w  -->  k x n
    thList = ['10', '20', '30', '40', '50', '60', '70', '80', '90', '95', '99']
    print('... loaded!\nadding images now ...')

    metric = Metrics()
    for i in range(len(modelList)):
        for j in range(len(modelList[i])):
            gt = gtList[i].reshape((gtList[i].shape[0], -1))
            pred = np.array(modelList[i][j])
            pred = pred.reshape((pred.shape[0], -1))
            score = thList[j]
            pred = pred[:min(args.maxprop, pred.shape[0])]
            print('adding im=%02d th=%02d' % (i+1,j+1))
            metric.add(gt, pred, score)

    ap3, pr3, rec3 = metric.AP(0.3)
    ap5, pr5, rec5 = metric.AP(0.5)
    abo, covering = metric.ABO()
    print('\n------------------------------------------')
    print('Time taken: %.2f secs' % (time.time()-startTime))
    print('Max predicted proposals per image allowed: %d' % args.maxprop)
    print('Model Name: %s' % args.mpath)
    print('GT Name: %s' % args.gtpath)
    print('\nAP at .3: %.2f' % (100*ap3))
    print('prec: ', [round(100*i, 2) for i in pr3])
    print('rec: ', [round(100*i, 2) for i in rec3])
    print('\nAP at .5: %.2f' % (100*ap5))
    print('prec: ', [round(100*i, 2) for i in pr5])
    print('rec: ', [round(100*i, 2) for i in rec5])
    print('\nABO: ', [round(100*i, 2) for i in abo])
    print('\nCovering: ', [round(100*i, 2) for i in covering])
    print('------------------------------------------\n')
    # import IPython; IPython.embed()
