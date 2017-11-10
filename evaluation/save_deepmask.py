import subprocess
import numpy as np
import scipy
from scipy.misc import imresize, imread, imsave
import os
import sys
import scipy.ndimage

def iou(mask1, mask2):
    if np.sum(np.logical_or(mask1, mask2)) == 0:
        return 10000.0
    return np.sum(np.logical_and(mask1, mask2)) / np.sum(np.logical_or(mask1, mask2))

def NMS(sorted_masks, nms_tr = 0.4, firstn = 20):
        result = [sorted_masks[0]] #list of class small_window
        for new_full_size_msk in sorted_masks[1:]:
            size_new_msk = np.sum(new_full_size_msk)
            max_iou = 0
            for s in result:
                max_iou = max(max_iou, iou(s, new_full_size_msk))
            if max_iou < nms_tr and size_new_msk > 600:
                result.append(new_full_size_msk)
        if (firstn is not None) and len(result) > firstn:
            return result[:firstn]
        else:
            return result


def load_val_separate_msk(img_path = '/media/4tb/dian/validation/Images/users/fred960315/validation', \
                    mask_path = '/media/4tb/dian/validation/Masks/users/fred960315/validation'):
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

"""
This function call deepmask and read output from deepmask
output:
Score: top200
Masks: top200 mask corresponding to the score
"""
def call_deepmask(image_in):
    imsave("/home/fred/Desktop/bot_mask_tool/deepmask/img.jpg", image_in.astype(np.uint8))
    bashCommand = "cd /home/fred/Desktop/bot_mask_tool/deepmask && export CUDA_VISIBLE_DEVICES=0 && th computeProposals.lua ./pretrained/deepmask/ -img ./img.jpg"
    subprocess.call(['/bin/bash', '-i', '-c', bashCommand])
    #read data output by deepmask
    score_200 = np.load("/home/fred/Desktop/bot_mask_tool/deepmask/score2npy.npy")
    masks_200 = np.load("/home/fred/Desktop/bot_mask_tool/deepmask/masks2npy.npy")
    return score_200, masks_200

val_imgs, val_bks, val_msks = load_val_separate_msk()
"""
Build 12*(40*c*w*h) array
"""


score_thrs = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.95,0.99]
result = []
score_thr_first_n = 100
NMS_first_n = 20
count = 0
print("len of val_imgs is {}".format(len(val_imgs)))
for img in val_imgs:
    score_mask = []
    score_200, masks_200 = call_deepmask(img)
    for score_thr in score_thrs:
        masks = []
        for s, m in zip(score_200[:,0], masks_200):
            if s > score_thr:
                masks.append(m)
        ###########do NMS Here###########
        if len(masks) > score_thr_first_n:
            masks = masks[:score_thr_first_n]
        masks = NMS(masks, nms_tr = 0.3, firstn = 20)
        
        
        
        masks = np.array(masks)
        score_mask.append(masks)
    score_mask = np.array(score_mask)
    print("img {} done, score_mask shape {}".format(count, score_mask.shape))
    count += 1
    result.append(score_mask)

#save
np.save("./saveOutputs/deepmask_result_NMS.npy", result)
