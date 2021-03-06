"""
DeepMask forward pass

"""
from __future__ import division
import tensorflow as tf
from tensorflow.contrib.layers import xavier_initializer

import argparse
import numpy as np
import pickle
import os
from scipy.misc import imresize
from models import rebuild_network_full_conv, build_resnet50_network,\
                    rebuild_original_network, rebuild_resent18_bootstrap
from scipy.misc import imresize
    

half_input_size = 96

def reference(img):
    pass

def flat_diff(tensor1, tensor2):
        tensor1 = tensor1.flatten()
        tensor2 = tensor2.flatten()
        return np.sum(np.square(tensor1 - tensor2))
        
def iou(mask1, mask2):
    if np.sum(np.logical_or(mask1, mask2)) == 0:
        return 10000.0
    return np.sum(np.logical_and(mask1, mask2)) / np.sum(np.logical_or(mask1, mask2))
    
#comparable class
class small_window():   
    def __init__(self, scale, score, pos_x, pos_y,mask, stride = 16, canonical_window = None):
        self.canonical_window = canonical_window
        self.scale = scale 
        self.score = score
        self.pos_x = pos_x #index, not real position
        self.pos_y = pos_y #index, not real position
        self.mask = mask #thresholded binary msk
        self.stride = stride
        #set variance for bootstrapping
        self.variance = None
        self.weighted_variance = None
        self.mask_variance = None
    def __gt__(self, small_window2):
        return self.score > small_window2.score
    
    def set_weighted_variance(self, input_number):
        self.weighted_variance = input_number
    
    def set_variance(self, variance):
        self.variance = variance
        
    def score_map_help(self):
        real_x = int(half_input_size/self.scale + self.pos_x*self.stride/self.scale)
        real_y = int(half_input_size/self.scale + self.pos_y*self.stride/self.scale)
        return self.score, real_x, real_y
        
    def generate_full_size_msk(self, origin_img, highlight_bond = False):
        img_shape = np.array(origin_img.shape[:2])
        result = np.zeros(img_shape + [200,200])
        # import pdb; pdb.set_trace()
        real_x = int(half_input_size/self.scale + self.pos_x*self.stride/self.scale)
        real_y = int(half_input_size/self.scale + self.pos_y*self.stride/self.scale)
        half_window_size = int(half_input_size/self.scale)
        resized_msk = imresize(self.mask, [half_window_size*2, half_window_size*2])
        if highlight_bond:
            resized_msk[:,:3]=1
            resized_msk[:,-3:]=1
            resized_msk[:3,:]=1
            resized_msk[-3:,:]=1
        
        result[real_y-half_window_size:real_y+half_window_size, real_x-half_window_size:real_x+half_window_size] = \
                            resized_msk
        result = (result > 0).astype(np.uint8)
        return result[:-200,:-200]

def cut_img(img, center_x, center_y, half_size):
    return img[center_y-half_size:center_y+half_size, center_x-half_size:center_x + half_size,:]
    

class forward_pass():
    def __init__(self, model_path, model_type, stride = 16, debug = False, addBg=False, background_diff_w = False):
        self.scales = [2**(i*0.25 - 1.25) for i in range(7)]
        self.debug = debug
        if addBg:
            bg_ph = tf.placeholder(tf.float32, [None, 160, 160, 3])
        else:
            bg_ph = None
        if not addBg:
            if not debug:
                img_ph = tf.placeholder(tf.float32, [None, None, None, 3])
                self.model_out, self.sess = rebuild_network_full_conv(img_ph, model_path, model_type, background_ph=bg_ph)
            else:
                img_ph = tf.placeholder(tf.float32, [None, 192, 192, 3])
                self.model_out, self.sess = rebuild_original_network(img_ph, model_path, model_type, background_ph=bg_ph)
        else:
            img_ph = tf.placeholder(tf.float32, [None, 192, 192, 3])
            self.model_out, self.sess = rebuild_original_network(img_ph, model_path, model_type, background_ph=bg_ph,\
                                        background_diff_w = background_diff_w)
        # msk, score = self.model_out(np.zeros([224,224,3]))
        # print(msk)
        # print(score)
        
        #no need to filt score in same position, NMS will filter those masks
        self.sorted_small_windows = None
        self.img_in = None
        self.stride = stride

        self.addBg = addBg

    def score_heatmap(self):
        result_score_map = np.zeros_like(self.img_in[:,:,0]).astype(np.float32)
        for small_w in self.sorted_small_windows:
            score, real_x, real_y = small_w.score_map_help()
            result_score_map[real_y-2:real_y+1, real_x-2:real_x+1] = score
        return result_score_map
    
    def msk_cut_score_NMS(self, score_tr = 0.99, nms_tr = 0.4, NMSfirstn = 10, score_firstn = 100):
        obj_small_windows = []
        score_list = []
        for small_window in reversed(self.sorted_small_windows):
            obj_small_windows.append(small_window)
            score_list.append(small_window.score)
            if small_window.score < score_tr:
                break
        if len(obj_small_windows) > score_firstn:
            obj_small_windows = obj_small_windows[:score_firstn]
            score_list = score_list[:score_firstn]
        score_list = np.array(score_list)
        NMS_mask, NMS_index =  self.NMS(obj_small_windows, nms_tr = nms_tr, firstn = NMSfirstn)
        return NMS_mask, score_list[NMS_index]

    def NMS(self, sorted_small_windows, nms_tr = 0.4, firstn = None):
        result = [sorted_small_windows[0].generate_full_size_msk(self.img_in)] #list of class small_window
        index = [0]
        for i, w in zip(range(1, len(sorted_small_windows)),sorted_small_windows[1:]):
            new_full_size_msk = w.generate_full_size_msk(self.img_in)
            size_new_msk = np.sum(new_full_size_msk)
            max_iou = 0
            for s in result:
                max_iou = max(max_iou, iou(s, new_full_size_msk))
            if max_iou < nms_tr and size_new_msk > 600:
                result.append(new_full_size_msk)
                index.append(i)
        if (firstn is not None) and len(result) > firstn:
            return result[:firstn], index[:firstn]
        else:
            return result, index
    def compute_multi_scale_slicing_window(self, img_in, background = None, msk_thr = 0.6):
        #for fair comparison, stride is 16 as well
        self.img_in = img_in.copy()
        small_window_list = [] 

        if not self.addBg:
            background = None

        for scale in self.scales:
            # print("scale {}".format(scale))
            scaled_img = imresize(img_in, [int(img_in.shape[0]*scale),int(img_in.shape[1]*scale)])
            scaled_img_hw = scaled_img.shape[:2]
            w_range = (scaled_img_hw[1]-half_input_size*2)//self.stride + 1
            h_range = (scaled_img_hw[0]-half_input_size*2)//self.stride + 1
            #slice images
            slice_windows = []
            for i in range(w_range):
                for j in range(h_range):
                    cropped_img = cut_img(scaled_img, half_input_size + self.stride*i, half_input_size + self.stride*j, half_input_size)
                    slice_windows.append(cropped_img)
                    
            #if too many slicing window, cut them to barch before feeding into GPU
            batch_size = 201
            num_batchs = len(slice_windows)//batch_size
            residue = len(slice_windows)%batch_size
            # print(num_batchs)
            slice_windows = np.array(slice_windows)
            msks = []
            scores = []
            for i in range(num_batchs):
                # print('batch index {}'.format(i))
                backgrounds = np.stack([background for _ in range(batch_size)], axis=0)
                msks_batch, scores_batch = self.model_out(slice_windows[batch_size*i:batch_size*(i+1),:,:,:], background=backgrounds, batch = True)
                msks.append(msks_batch)
                scores.append(scores_batch)
            if residue != 0:
                backgrounds = np.stack([background for _ in range(len(slice_windows[batch_size*(num_batchs):,:,:,:]))], axis=0)
                msks_batch, scores_batch = self.model_out(slice_windows[batch_size*(num_batchs):,:,:,:], background=backgrounds, batch = True)
                msks.append(msks_batch)
                scores.append(scores_batch)
            msks = np.vstack(msks)
            scores = np.vstack(scores)
            # import pdb; pdb.set_trace()
            for i in range(w_range):
                for j in range(h_range):
                    canonical_window = slice_windows[i*h_range + j]
                    msk = msks[i*h_range + j][:,:,1]
                    msk = (msk > msk_thr).astype(np.uint8)#msk cut score threshold
                    score = scores[i*h_range + j].flatten()[-1]
                    temp = small_window(scale, score, i, j, msk, stride = self.stride, canonical_window = canonical_window)
                    small_window_list.append(temp)
            #sort small_window_list
        small_window_list.sort()
        self.sorted_small_windows = small_window_list
        return small_window_list
    def compute_multiscale_masks(self, img_in, background = None, msk_thr = 0.5):
        if self.debug:
            print("debug model, only support caonoical slicing window model")
            raise NotImplementedError
        self.img_in = img_in.copy()
        small_window_list = []
        for scale in self.scales:
            # print("scale {}".format(scale))
            scaled_img = imresize(img_in, [int(img_in.shape[0]*scale),int(img_in.shape[1]*scale)])
            scaled_img_hw = scaled_img.shape[:2]
            msk, score = self.model_out(scaled_img, background)
            msk = msk[:,:,:,1] #prob of 1
            msk = (msk > msk_thr).astype(np.uint8)#msk cut score threshold
            score = score[:,:,:,1] #score
            score_h, score_w = score.shape[1:3]
            # import pdb; pdb.set_trace()
            # assert(score_w == (scaled_img_hw[1]-half_input_size*2)//self.stride + 1)
            # assert(score_h == (scaled_img_hw[0]-half_input_size*2)//self.stride + 1)
            for i in range(score_h):
                for j in range(score_w):
                    h = 96 + i*16
                    w = 96 + j*16
                    canonical_window = scaled_img[h-96:h+96, w-96:w+96,:]
                    temp = small_window(scale, score[0,i,j], j, i, msk[j + i*score_w,:,:], canonical_window = canonical_window)
                    small_window_list.append(temp)
        #sort small_window_list
        small_window_list.sort()
        self.sorted_small_windows = small_window_list
        return small_window_list
    #top_n mask without nms
    def top_n_masks(self, top_n = 10):
        full_size_msks = []
        scores = []
        for small_window in reversed(self.sorted_small_windows[-top_n:]):
            scores.append(small_window.score)
            full_size_msks.append(small_window.generate_full_size_msk(self.img_in, highlight_bond = True))
        return full_size_msks, scores
    
    def destory_graph(self):
        tf.reset_default_graph()
        self.sess.close()
    
    def clean_result(self):
        self.sorted_small_windows = None
        self.img_in = None
        return

class forward_pass_bootstrap(forward_pass): #current only resnet18 bootstrap
    def __init__(self, model_path, stride = 16, debug = False, addBg=False,\
                    background_diff_w = False, num_heads = 5):
        assert (debug is False)
        self.scales = [2**(i*0.25 - 1.25) for i in range(7)]
        self.debug = debug
        if addBg:
            bg_ph = tf.placeholder(tf.float32, [None, 160, 160, 3])
        else:
            bg_ph = None
        if not addBg:
            img_ph = tf.placeholder(tf.float32, [None, None, None, 3])
            self.model_out, self.sess = rebuild_resent18_bootstrap(img_ph, model_path,\
                                background_ph=bg_ph, background_diff_w = background_diff_w,\
                                num_heads = num_heads)
        else:
            img_ph = tf.placeholder(tf.float32, [None, 192, 192, 3])
            self.model_out, self.sess = rebuild_resent18_bootstrap(img_ph, model_path,\
                                background=bg_ph, model_path = model_path, background_diff_w = background_diff_w, num_heads = num_heads)
        # msk, score = self.model_out(np.zeros([224,224,3]))
        # print(msk)
        # print(score)
        
        #no need to filt score in same position, NMS will filter those masks
        self.sorted_small_windows = None
        self.sorted_small_windows_list = None
        self.img_in = None
        self.stride = stride
        self.num_heads = num_heads
        self.addBg = addBg
    
    def compute_multi_scale_slicing_window(self, img_in, background = None, msk_thr = 0.6):
        raise NotImplementedError
    
    def compute_multiscale_masks(self, img_in, background = None, msk_thr = 0.6):
        if self.debug:
            print("debug model, only support caonoical slicing window model")
            raise NotImplementedError
        self.img_in = img_in.copy()
        small_window_list = [[] for i in range(self.num_heads)]
        for scale in self.scales:
            # print("scale {}".format(scale))
            scaled_img = imresize(img_in, [int(img_in.shape[0]*scale),int(img_in.shape[1]*scale)])
            scaled_img_hw = scaled_img.shape[:2]
            msks_scores = self.model_out(scaled_img, background)
            count = 0
            for msk, score in msks_scores:
                msk = msk[:,:,:,1] #prob of 1
                msk = (msk > msk_thr).astype(np.uint8)#msk cut score threshold
                score = score[:,:,:,1] #score
                score_h, score_w = score.shape[1:3]
                # import pdb; pdb.set_trace()
                # assert(score_w == (scaled_img_hw[1]-half_input_size*2)//self.stride + 1)
                # assert(score_h == (scaled_img_hw[0]-half_input_size*2)//self.stride + 1)
                for i in range(score_h):
                    for j in range(score_w):
                        h = 96 + i*16
                        w = 96 + j*16
                        canonical_window = scaled_img[h-96:h+96, w-96:w+96,:]
                        if not (canonical_window.shape[0] == 192 and canonical_window.shape[1] == 192):
                            continue
                        assert(canonical_window.shape[0] == 192 and canonical_window.shape[1] == 192)
                        temp = small_window(scale, score[0,i,j], j, i, msk[j + i*score_w,:,:], canonical_window = canonical_window)
                        small_window_list[count].append(temp)
                count += 1
        #compute the variance
        num_slicing_window = len(small_window_list[0])
        for i in range(num_slicing_window):
            scores = []
            masks = []
            for j in range(self.num_heads):
                scores.append(small_window_list[j][i].score)
                masks.append(small_window_list[j][i].mask)
            var = np.var(np.array(scores))
            masks = np.array(masks)
            #compute variance of masks and normalize masks
            mean_mask_size = np.sum(masks)/(1.0 * self.num_heads)
            mask_var = np.var(masks, 0)
            normalized_mask_var = np.sum(mask_var)/mean_mask_size
            for j in range(self.num_heads):
                small_window_list[j][i].set_variance(var)
                small_window_list[j][i].set_weighted_variance(var * np.exp(np.mean(scores)))
                #scale msk variance by score
                small_window_list[j][i].mask_variance = normalized_mask_var*np.exp(np.mean(scores))
        #sort small_window_list
        for i in range(self.num_heads):
            small_window_list[i].sort()
        self.sorted_small_windows_list = small_window_list
        return small_window_list
    def topn_mask_var(self, topn = 5):
        top_mask_var = self.sorted_small_windows_list.copy()
        for i in range(self.num_heads):
            top_mask_var[i].sort(key = lambda a: a.mask_variance)
        top_mask_var = np.array(top_mask_var)
        return top_mask_var[:,-topn:].T
    
    def topn_var(self, topn = 5):
        top_var = self.sorted_small_windows_list.copy()
        for i in range(self.num_heads):
            top_var[i].sort(key = lambda a: a.variance)
        top_var = np.array(top_var)
        return top_var[:,-topn:].T
    def weighted_topn_var(self, topn = 5):
        top_weighted_var = self.sorted_small_windows_list.copy()
        for i in range(self.num_heads):
            top_weighted_var[i].sort(key = lambda a: a.weighted_variance)
        top_weighted_var = np.array(top_weighted_var)
        return top_weighted_var[:,-topn:].T
        
        
    def score_heatmap_single_head(self, head_index):
        assert(head_index >= 0 and head_index < self.num_heads)
        self.sorted_small_windows = self.sorted_small_windows_list[head_index]
        return self.score_heatmap()
    def score_variance_map(self):
        score_heatmaps = []
        for i in range(self.num_heads):
            score_heatmaps.append(self.score_heatmap_single_head(i))
        score_heatmaps = np.array(score_heatmaps)
        # import pdb; pdb.set_trace()
        result_score_variance_map = np.var(score_heatmaps, axis = 0)
        return result_score_variance_map/(np.max(result_score_variance_map))
    def msk_cut_score_NMS_head(self, head_index, score_tr = 0.99, nms_tr = 0.4, NMSfirstn = 10,\
                                score_firstn = 100):
        self.sorted_small_windows = self.sorted_small_windows_list[head_index]
        return self.msk_cut_score_NMS(score_tr = score_tr, nms_tr = nms_tr, \
                                NMSfirstn = NMSfirstn, score_firstn = score_firstn)
                                    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model_path', type=str)
    args = parser.parse_args()
    
    debug(args.model_path)
