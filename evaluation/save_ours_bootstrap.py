import scipy.ndimage
import numpy as np
import time
import pickle
from scipy.misc import imresize, imread, imsave
import os
import sys
sys.path.insert(0, os.path.relpath("../"))
from inference import *
from scipy.misc import imresize
import matplotlib.pyplot as plt
import subprocess


def load_val_separate_msk(img_path = './valimages', \
                    mask_path = './valmasks'):
    img_items = sorted([img_path + '/'+ item for item in os.listdir(img_path)])
    mask_items = sorted([mask_path + '/'+ item for item in os.listdir(mask_path)])
    imgs = np.zeros((len(img_items), 350, 430, 3))
    masks = []
    backgrounds = np.zeros((len(img_items), 160, 160, 3))
    for i, img_name in enumerate(img_items):
        key = img_name.split('.jpg')[0].split('/')[-1]
        imgs[i] = imread(img_name)[20:370,230:660,:]
        backgrounds[i] = imresize(imread(img_name)[100:400, 250:600], [160,160])
        #also prepare data for deepmask
        dp_mask_img = imread(img_name)[20:370,230:660,:]

        mask_names = list(filter(lambda s: '/'+key+'_mask' in s, mask_items))
        diff_masks = []
        for j, mask_name in enumerate(mask_names):
            mask = (imread(mask_name).sum(axis=2) > 0).astype(np.uint8)[20:370,230:660]
            if i == 1 and j == 2:
                tmp1 = np.zeros_like(mask)
                tmp2 = np.zeros_like(mask)
                tmp1[:,:180] = mask[:,:180]
                tmp2[:,180:] = mask[:,180:]
                tmp2 = scipy.ndimage.morphology.binary_fill_holes(tmp2)
                diff_masks.append(tmp1)
                diff_masks.append(tmp2)
            else:
                diff_masks.append(mask)
        diff_masks = np.array(diff_masks)
        masks.append(diff_masks)
    # index_list = [36,38,4,10,18,19,28,34]
    order = [0,11,22,33,36,\
            1,2,37,38,39,\
            3,4,5,6,7,\
            8,9,10,12,13,\
            14,15,16,17,18,\
            19,20,21,23,24,\
            25,26,27,28,29,\
            30,31,32,34,35]
    masks = np.array(masks)
    return imgs.astype(np.uint8)[order,:,:,:], \
                    backgrounds.astype(np.uint8)[order,:,:,:],\
                    masks[order]

val_imgs, val_bks, val_masks = load_val_separate_msk()

#wrapper for bootstrapping code 40 x num_heads x c x w x h
def eval_wrapper(model_name, msk_thr, num_heads = 5, nms_thr=0.3, model_type = "resnet18"):
    fw_pass = forward_pass_bootstrap(model_name, num_heads = num_heads)
    output = []
    count = 0
    for img, background in zip(val_imgs, val_bks):
        count += 1
        print('Evaluating: {} img {}'.format(model_name[:13],count))
        img_in = np.lib.pad(img, ([100,100],[100,100],[0,0]), 'constant', constant_values=127)
        slice_windows_increase_order = fw_pass.compute_multiscale_masks(img_in, msk_thr = msk_thr)
        
        heads_outputs = []
        for i in range(num_heads):
            NMS_mask, NMS_score = fw_pass.msk_cut_score_NMS_head(head_index = i, score_tr = 0.7,\
                                                                            nms_tr = nms_thr, NMSfirstn = 100,\
                                                                            score_firstn = 100)
            NMS_mask = np.array(NMS_mask)
            NMS_mask = NMS_mask[:,100:,100:]
            NMS_mask = NMS_mask[:,:-100,:-100]
            NMS_mask  = np.reshape(NMS_mask, [NMS_mask.shape[0], -1])
            assert(NMS_mask.shape[0] == NMS_score.shape[0])
            NMS_score = np.reshape(NMS_score, [1,-1])
        
            heads_outputs.append([NMS_score, NMS_mask])
        # import pdb; pdb.set_trace()
        output.append(heads_outputs)
    fw_pass.destory_graph()
    return output
import glob

# model = glob.glob("../models/501*199000.meta")[0][:-5]
model = glob.glob("../models/503*199000.meta")[0][:-5]
print(model)
num_heads = 5
for msk_thr in [0.5,0.6,0.7]:
    output = eval_wrapper(model, msk_thr, num_heads = num_heads)
    for j in range(num_heads):
        single_head_out = []
        for i in range(val_imgs.shape[0]):
            import pdb; pdb.set_trace()
            single_head_out.append(output[i][j])
        np.save("./savedOutputs/%s_msk_thr%.2f_head_%d.npy"%(model[10:][:3], msk_thr, j), single_head_out)


