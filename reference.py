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
from models import rebuild_network, build_network



# class multi_scale_reference():
#     def __init__(self, model_path):
#         self.img_ph, self.score_tensor, self.msk_tensor = load_rebuild(model_path)
#         self.mask = None
#         self.score = None
#     def 
#     def multi_scale_forward_path(self, img_in):
#         feed_dict = {self.img_ph : img_in}
#         return self.sess.run([self.score_tensor, self.msk_tensor], feed_dict = feed_dict)
    
#     def top_n(self, top_n):
#         pass
#     def 
    
    

def reference(img):
    pass

def flat_diff(tensor1, tensor2):
        tensor1 = tensor1.flatten()
        tensor2 = tensor2.flatten()
        return np.sum(np.square(tensor1 - tensor2))

def debug(model_path):
    """
    sanity check, check if fully-convolutionary version exactly replicate the behavior of
    the originnal network
    
    input: 224*240*3 img (2x224*224*3 for FC network)
    """
    sess = tf.Session()
    test_input = (np.random.rand(224,240,3)*255).astype(np.float32) #img format
    test_input[:,:,:] = 127.5
    # test_input[:,200:,:] = 127.5
    test1 = (test_input[:,:224,:].copy()/255.0)-0.5
    test2 = (test_input[:,16:,:].copy()/255.0)-0.5
    
    #original model
    
    img_ph = tf.placeholder(tf.float32, [1, 224, 224, 3])
    mask, score, shared_trunk, vgg_out = build_network(img_ph, dropout = 1, debug = True)
    
    model_saver = tf.train.Saver()
    model_saver.restore(sess, model_path)
    
    mask1, score1, shared_truck1, vgg_out1 = sess.run([mask, score, shared_trunk,vgg_out], feed_dict = {img_ph:[test1]})
    mask2, score2, shared_truck2 = sess.run([mask, score, shared_trunk], feed_dict = {img_ph:[test2]})
    
    print(score1)
    print(score2)
    
    sess.close()
    tf.reset_default_graph()
    
    img_ph = tf.placeholder(tf.float32, [1, None, None, 3])
    model_out, sess = rebuild_network(img_ph, model_path, debug = True)
    conv_mask, conv_score, conv_shared_truck, conv_vgg_out = model_out(test_input[:,:224,:].copy())
    print(flat_diff(conv_vgg_out[:,:14,:14,:], vgg_out1))
    print(flat_diff(shared_truck1, conv_shared_truck))
    print(flat_diff(conv_mask, mask1))
    print(flat_diff(conv_score, score1))
    
    print(conv_score)
    print(score1)
    
    conv_mask, conv_score, conv_shared_truck, conv_vgg_out = model_out(test_input.copy())
    print("diff vgg out first slice")
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
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model_path', type=str)
    args = parser.parse_args()
    
    debug(args.model_path)
    
    # img_ph = tf.placeholder(tf.float32, [1, None, None, 3])
    # model_out, sess = rebuild_network(img_ph, args.model_path)
    # msk_result, score_result = model_out(np.zeros([224,240,3]))
    # print(msk_result.shape)
    # print(score_result.shape)
    import pdb; pdb.set_trace()
        
        
        

if __name__ == '__main__':
   main()

