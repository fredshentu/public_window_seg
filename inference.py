"""
DeepMask reference

"""

import tensorflow as tf
from tensorflow.contrib.layers import xavier_initializer

import argparse
import numpy as np
import pickle
import os
from scipy.misc import imresize
from models import rebuild_network, build_resnet50_network,rebuild_original_network
from scipy.misc import imresize
    

half_input_size = 80

def reference(img):
    pass

def flat_diff(tensor1, tensor2):
        tensor1 = tensor1.flatten()
        tensor2 = tensor2.flatten()
        return np.sum(np.square(tensor1 - tensor2))
        
def iou(mask1, mask2):
    return np.sum(np.logical_and(mask1, mask2)) / np.sum(np.logical_or(mask1, mask2))
    
def debug(model_path):
    """
    sanity check, check if fully-convolutionary version exactly replicate the behavior of
    the originnal network
    
    input: half_input_size*2*240*3 img (2x224*224*3 for FC network)
    """
    sess = tf.Session()
    test_input = (np.random.rand(224,240,3)*255).astype(np.float32) #img format
    test_input[:,:,:] = 127.5
    # test_input[:,200:,:] = 127.5
    test1 = (test_input[:,:224,:].copy()/255.0)-0.5
    test2 = (test_input[:,16:,:].copy()/255.0)-0.5
    
    #original model
    
    img_ph = tf.placeholder(tf.float32, [1, 224, 224, 3])
    mask, score, shared_trunk, vgg_out = build_resnet50_network(img_ph, sess, debug = True)
    
    model_saver = tf.train.Saver()
    model_saver.restore(sess, model_path)
    
    mask1, score1, shared_truck1, vgg_out1 = sess.run([mask, score, shared_trunk,vgg_out], feed_dict = {img_ph:[test1]})
    mask2, score2, shared_truck2 = sess.run([mask, score, shared_trunk], feed_dict = {img_ph:[test2]})
    
    print(score1)
    print(score2)
    
    sess.close()
    tf.reset_default_graph()
    
    img_ph = tf.placeholder(tf.float32, [1, None, None, 3])
    model_out, sess = rebuild_network(img_ph, model_path, "resnet50",debug = True)
    conv_mask, conv_score, conv_shared_truck, conv_vgg_out = model_out(test_input[:,:224,:].copy())
    print(flat_diff(conv_vgg_out[:,:14,:14,:], vgg_out1))
    print(flat_diff(shared_truck1, conv_shared_truck))
    print(flat_diff(conv_mask, mask1))
    print(flat_diff(conv_score, score1))
    
    print(conv_score)
    print(score1)
    test_input = (np.random.rand(224,250,3)*255).astype(np.float32)
    conv_mask, conv_score, conv_shared_truck, conv_vgg_out = model_out(test_input)
    print("resnet output size")
    print(conv_vgg_out.shape)
    print(flat_diff(conv_vgg_out[:,:,:14,:], vgg_out1))
    
    exit()
    conv_mask1 = conv_mask[0]
    conv_score1 = conv_score[:,:,0,:]
    conv_shared_truck1 = conv_shared_truck[:,:,0,:]
    
    conv_mask2 = conv_mask[1]
    conv_score2 = conv_score[:,:,1,:]
    conv_shared_truck2 = conv_shared_truck[:,:,1,:]
        
    print("Start test......")
    print("Assert share_truck diff1 {} diff2 {}".format(flat_diff(conv_shared_truck1, shared_truck1),\
                                                        flat_diff(conv_shared_truck2, shared_truck2)))
    print("Assert msk diff1 {} diff2 {}".format(flat_diff(conv_mask1, mask1), flat_diff(conv_mask2, mask2)))
    print("Assert score diff1 {} diff2 {}".format(flat_diff(score1, conv_score1), flat_diff(score2, conv_score2)))
    print("all diff should be very very small")
    pass

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
    def __gt__(self, small_window2):
        return self.score > small_window2.score
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
        
        return result[:-200,:-200]

def cut_img(img, center_x, center_y, half_size):
    return img[center_y-half_size:center_y+half_size, center_x-half_size:center_x + half_size,:]

class forward_pass():
    def __init__(self, model_path, model_type, stride = 16, debug = False):
        self.scales = [2**(i*0.25 - 0.5) for i in range(7)]
        
        if not debug:
            img_ph = tf.placeholder(tf.float32, [None, None, None, 3])
            self.model_out, self.sess = rebuild_network(img_ph, model_path, model_type)
        else:
            img_ph = tf.placeholder(tf.float32, [None, 160, 160, 3])
            self.model_out, self.sess = rebuild_original_network(img_ph, model_path, model_type)
        # msk, score = self.model_out(np.zeros([224,224,3]))
        # print(msk)
        # print(score)
        
        #no need to filt score in same position, NMS will filter those masks
        self.sorted_small_windows = None
        self.img_in = None
        self.stride = stride
    def msk_cut_score_NMS(self, score_tr = 0.99, nms_tr = 0.4):
        obj_small_windows = []
        for small_window in reversed(self.sorted_small_windows):
            obj_small_windows.append(small_window)
            if small_window.score < score_tr:
                break
        return self.NMS(obj_small_windows, nms_tr = nms_tr)
        
    def NMS(self, sorted_small_windows, nms_tr = 0.4):
        result = [sorted_small_windows[0].generate_full_size_msk(self.img_in)] #list of class small_window
        for w in sorted_small_windows[1:]:
            new_full_size_msk = w.generate_full_size_msk(self.img_in)
            max_iou = 0
            for s in result:
                max_iou = max(max_iou, iou(s, new_full_size_msk))
            if max_iou < nms_tr:
                result.append(new_full_size_msk)
        return result
    def compute_multi_scale_slicing_window(self, img_in, msk_thr = 0.5):
        #for fair comparison, stride is 16 as well
        self.img_in = img_in.copy()
        small_window_list = []
        for scale in self.scales:
            
            print("scale {}".format(scale))
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
                # import pdb; pdb.set_trace()
                msks_batch, scores_batch = self.model_out(slice_windows[batch_size*i:batch_size*(i+1),:,:,:], batch = True)
                msks.append(msks_batch)
                scores.append(scores_batch)
            if residue != 0:
                msks_batch, scores_batch = self.model_out(slice_windows[batch_size*(num_batchs):,:,:,:], batch = True)
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
    def compute_multiscale_masks(self, img_in, msk_thr = 0.5):
        self.img_in = img_in.copy()
        small_window_list = []
        for scale in self.scales:
            print("scale {}".format(scale))
            scaled_img = imresize(img_in, [int(img_in.shape[0]*scale),int(img_in.shape[1]*scale)])
            scaled_img_hw = scaled_img.shape[:2]
            msk, score = self.model_out(scaled_img)
            msk = msk[:,:,:,1] #prob of 1
            msk = (msk > msk_thr).astype(np.uint8)#msk cut score threshold
            score = score[:,:,:,1] #score
            score_h, score_w = score.shape[1:3]
            # import pdb; pdb.set_trace()
            # assert(score_w == (scaled_img_hw[1]-half_input_size*2)//self.stride + 1)
            # assert(score_h == (scaled_img_hw[0]-half_input_size*2)//self.stride + 1)
            for i in range(score_h):
                for j in range(score_w):
                    temp = small_window(scale, score[0,i,j], j, i, msk[j + i*score_w,:,:])
                    small_window_list.append(temp)
        #sort small_window_list
        small_window_list.sort()
        self.sorted_small_windows = small_window_list
        return 
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
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model_path', type=str)
    args = parser.parse_args()
    
    debug(args.model_path)