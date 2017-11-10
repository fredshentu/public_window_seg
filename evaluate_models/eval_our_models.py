import scipy.ndimage
import numpy as np
import time
import pickle
from scipy.misc import imresize, imread, imsave
import sys
sys.path.append('/home/fred/Desktop/window_seg/')
from inference import *
from scipy.misc import imresize
import matplotlib.pyplot as plt
import subprocess
import os

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
# import pdb; pdb.set_trace()
# np.save("ground_truth_msk", val_masks)
score_thrs = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.95,0.99]
def eval_wrapper(model_name, nms_thr=0.3, addBg = False,\
                                     bk_diff_w = False, model_type = "resnet18"):
    fw_pass = forward_pass(model_name, model_type, addBg = addBg, background_diff_w = bk_diff_w,debug = False)
    output = []
    count = 0
    for img, background in zip(val_imgs, val_bks):
        img_out = []
        count += 1
        print('Evaluating: {} img {}'.format(model_name[:13],count))
        img_in = np.lib.pad(img, ([100,100],[100,100],[0,0]), 'constant', constant_values=127)
        if not addBg:
#             print("fully conv")
            slice_windows_increase_order = fw_pass.compute_multiscale_masks(img_in, msk_thr = 0.6)
        else:
#             print("stupid slicing window")
            slice_windows_increase_order = fw_pass.compute_multi_scale_slicing_window(img_in, msk_thr = 0.6,\
                                                                                        background = background)
        for score_thr in score_thrs:
            print("Evaluating score threshold {}".format(score_thr))
            score_cut_NMS = np.array(fw_pass.msk_cut_score_NMS(score_tr = score_thr, nms_tr=nms_thr,\
                                                       NMSfirstn = 20, score_firstn = 100))
            score_cut_NMS = score_cut_NMS[:,100:,100:]
            score_cut_NMS = score_cut_NMS[:,:-100,:-100]
            score_cut_NMS = np.array(score_cut_NMS)
            img_out.append(score_cut_NMS)
        output.append(img_out)
    fw_pass.destory_graph()
    return output
import glob






# modelGroup1 = [glob.glob("../models/023*199000*.meta")[0][:-5],\
#                 glob.glob("../models/022*199000*.meta")[0][:-5],\
#                 glob.glob("../models/021*199000*.meta")[0][:-5],\
#                 glob.glob("../models/020*199000*.meta")[0][:-5],\
#                 glob.glob("../models/019*199000*.meta")[0][:-5],
#                 ] #no bk

# for model in modelGroup1:
#     print("Loading model %s"%model)
#     output = eval_wrapper(model, addBg = False, bk_diff_w = True)
#     np.save("./result_data/%s.npy"%model[:10][:3], output)


# modelGroup2 = [glob.glob("../models/015*184000*.meta")[0][:-5],\
#                 glob.glob("../models/016*184000*.meta")[0][:-5],\
#                 glob.glob("../models/017*184000*.meta")[0][:-5],\
#                 glob.glob("../models/018*199000*.meta")[0][:-5],\
#                 ] #bk share weights

# for model in modelGroup2:
#     print("Loading model %s"%model)
#     output = eval_wrapper(model, addBg = True, bk_diff_w = False)
#     np.save("./result_data/%s.npy"%model[:10][:3], output)

modelGroup3 = [glob.glob("../models/011*199000*.meta")[0][:-5],\
                glob.glob("../models/012*199000*.meta")[0][:-5],\
                glob.glob("../models/013*199000*.meta")[0][:-5],\
                glob.glob("../models/014*199000*.meta")[0][:-5],\
                ] #bk unshare weights

for model in modelGroup3:
    print("Loading model %s"%model)
    output = eval_wrapper(model, addBg = True, bk_diff_w = True)
    np.save("./result_data/%s.npy"%model[:10][:3], output)
    