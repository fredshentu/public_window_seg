from __future__ import print_function, division
import numpy as np

class Metrics(object):
    def __init__(self, maskTh=None):
        self.maskTh = maskTh
        self.proposals = None
        self.numIm = 0
        self.numGtProps = 0
        self.maxGtPropPerIm = 0
        self.abodata = None

    def binaryIoU(self, m1, m2):
        if np.sum(np.logical_or(m1, m2)) == 0:
            return 1.0
        return np.sum(np.logical_and(m1, m2)) / np.sum(np.logical_or(m1, m2))

    def add(self, gt, pred, score, imId):
        pred = pred if self.maskTh is None else (pred>self.maskTh).astype(int)
        iou = np.ones((gt.shape[0], pred.shape[0]), dtype=np.float)
        for i in range(gt.shape[0]):
            for j in range(pred.shape[0]):
                iou[i,j] = self.binaryIoU(gt[i], pred[j])

        imStat = np.stack((np.ones((pred.shape[0],))*imId,
            np.argmax(iou, axis=0), np.amax(iou, axis=0), score), axis=-1)
        if self.proposals is None:
            self.proposals = imStat
        else:
            self.proposals = np.concatenate((self.proposals, imStat))
        self.numIm = max(self.numIm, imId+1)
        self.numGtProps += gt.shape[0]
        self.maxGtPropPerIm = max(self.maxGtPropPerIm, gt.shape[0])

        aboStat = np.stack((np.amax(iou, axis=1), np.sum(gt, axis=1),
            np.argmax(iou, axis=1)), axis=-1)
        if self.abodata is None:
            self.abodata = aboStat
        else:
            self.abodata = np.concatenate((self.abodata, aboStat))

    def _voc_ap(self, rec, prec, use_07_metric=False):
        if use_07_metric:
            # 11 point metric
            ap = 0.
            for t in np.arange(0., 1.1, 0.1):
                if np.sum(rec >= t) == 0:
                    p = 0
                else:
                    p = np.max(prec[rec >= t])
                ap = ap + p / 11.
        else:
            # correct AP calculation
            # first append sentinel values at the end
            mrec = np.concatenate(([0.], rec, [1.]))
            mpre = np.concatenate(([0.], prec, [0.]))

            # compute the precision envelope
            for i in range(mpre.size - 1, 0, -1):
                mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

            # to calculate area under PR curve, look for points
            # where X axis (recall) changes value
            i = np.where(mrec[1:] != mrec[:-1])[0]

            # and sum (\Delta recall) * prec
            ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
        return ap

    def AP(self, iouTh=0.5):
        self.proposals = self.proposals[(-self.proposals[:,-1]).argsort()]
        imMat = np.zeros((self.numIm, self.maxGtPropPerIm), dtype=int)
        nd = self.proposals.shape[0]
        tp = np.zeros(nd)
        fp = np.zeros(nd)
        for i in range(nd):
            imId = int(self.proposals[i][0])
            propId = int(self.proposals[i][1])
            maxiou = self.proposals[i][2]
            if maxiou > iouTh and not imMat[imId][propId]:
                tp[i] = 1
                imMat[imId][propId] = 1
            else:
                fp[i] = 1
        tp = np.cumsum(tp)
        fp = np.cumsum(fp)
        rec = tp / float(self.numGtProps)
        prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
        ap = self._voc_ap(rec, prec, True)
        return ap, prec, rec

    def ABO(self):
        abo = np.sum(self.abodata[:,0]) / self.abodata.shape[0]
        covering = np.sum(self.abodata[:,0] * self.abodata[:,1]) / np.sum(
            self.abodata[:,1])
        maxPredProp = np.max(self.abodata[:,2]) + 1
        return abo, covering, maxPredProp


if __name__ == '__main__':
    import argparse
    import time
    startTime = time.time()
    parser = argparse.ArgumentParser(description="Metric Evaluator")
    parser.add_argument('-gp', '--gtpath', type=str, help="gt path",
        default="./savedOutputs/ground_truth_msk.npy")
    parser.add_argument('-mp', '--mpath', type=str, help="model path")
    parser.add_argument('-k', '--maxprop', type=int, default=None,
        help="max limit on number of predicted proposals per image")
    parser.add_argument('-th', '--scoreTh', type=float, default=None,
        help="discard proposals per image below this threshold")
    args = parser.parse_args()

    print('loading models ...')
    '''
    1) Do NMS over top 100 proposals per image to output remaining "k" proposals
    2) Save format as follows:
        -> list of 40 elements, where
        -> each element is a list of two things:
        -> [score => np.array((1,k), dtype=float),
        ->      masks=>np.array((k,n), dtype=uint8)] where n=h*w
    '''
    gtList = np.load(args.gtpath)
    modelList = np.load(args.mpath)
    print('... loaded!\nadding images now ...')

    metric = Metrics()
    for i in range(len(modelList)):
        # if i>=30 and i<=34: continue
        gt = gtList[i][1]
        pred = modelList[i][1]
        score = modelList[i][0].flatten()
        if args.scoreTh is not None:
            topK = np.where(score > args.scoreTh)
            score = score[topK]
            pred = pred[topK]
        if args.maxprop is not None:
            topK = min(args.maxprop, pred.shape[0])
            score = score[:topK]
            pred = pred[:topK]
        print('adding im=%02d' % (i+1))
        metric.add(gt, pred, score, i)

    ap3, pr3, rec3 = metric.AP(0.3)
    ap5, pr5, rec5 = metric.AP(0.5)
    abo, cov, maxPredProp = metric.ABO()
    maxprop = 100 if args.maxprop is None else args.maxprop
    print('\n------------------------------------------')
    print('Time taken: %.2f secs' % (time.time()-startTime))
    print('Max predicted proposals per image allowed: %d' % maxprop)
    print('Score threshold for proposals per image: %s' % str(args.scoreTh))
    print('Model Name: %s' % args.mpath)
    print('GT Name: %s' % args.gtpath)
    print('\nAP at .3: %.2f' % (100*ap3))
    # print('prec: ', [round(100*i, 2) for i in pr3])
    # print('rec: ', [round(100*i, 2) for i in rec3])
    print('\nAP at .5: %.2f' % (100*ap5))
    # print('prec: ', [round(100*i, 2) for i in pr5])
    # print('rec: ', [round(100*i, 2) for i in rec5])
    print('\nABO: %.2f' % (100*abo))
    print('Covering: %.2f' % (100*cov))
    print('MaxPredProposalPerImage: %d\n' % maxPredProp)
    print([round(100*i, 2) for i in [ap3, ap5, abo, cov]] + [int(maxPredProp)])
    print('------------------------------------------\n')
    # import IPython; IPython.embed()
