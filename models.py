'''
Utils for VGG netoworks and seg/scoring network
'''


import tensorflow as tf
from tensorflow.contrib.layers import xavier_initializer

import argparse
import numpy as np
import pickle
from resnet import resnet_v1
slim = tf.contrib.slim
import sys


    
with open('./data/vgg16.pkl', 'rb') as f:
        if sys.version_info > (3, 0):
            net_data = pickle.load(f, encoding='latin1')
        else:
            net_data = pickle.load(f)

def var(name, data, trainable):
    return tf.get_variable(name, initializer=tf.constant(data),trainable=trainable)

def new_var(name, shape):
    return tf.get_variable(name, shape=shape, initializer=xavier_initializer())

def conv(input, kernel, biases, k_h, k_w, c_o, s_h, s_w,  padding="SAME", group=1):
    '''From https://github.com/ethereon/caffe-tensorflow
    '''
    c_i = input.get_shape()[-1]
    assert c_i%group==0
    assert c_o%group==0
    convolve = lambda i, k: tf.nn.conv2d(i, k, [1, s_h, s_w, 1], padding=padding)

    if group==1:
        conv = convolve(input, kernel)
    else:
        input_groups = tf.split(3, group, input)
        kernel_groups = tf.split(3, group, kernel)
        output_groups = [convolve(i, k) for i,k in zip(input_groups, kernel_groups)]
        conv = tf.concat(3, output_groups)
    return tf.nn.bias_add(conv, biases)

def resnet_50_network(img_ph, is_training=False, reuse=False, scope=None):
    with slim.arg_scope(resnet_v1.resnet_arg_scope()):
      _, end_points = resnet_v1.resnet_v1_50(img_ph, reuse=reuse, is_training=is_training)
    x = end_points['resnet_v1_50/block3/unit_5/bottleneck_v1']
    return x


def resnet_18_network(img_ph, is_training=False, reuse=False, scope=None):
    with slim.arg_scope(resnet_v1.resnet_arg_scope()):
      _, end_points = resnet_v1.resnet_v1_18(img_ph, reuse=reuse, is_training=is_training)
    x = end_points['resnet_v1_18/block3/unit_1/bottleneck_v1']
    return x


def vgg_network(img_ph, trainable=False, reuse=False):
    with tf.variable_scope('vgg', reuse=reuse) as sc:
        x = img_ph
        # Collect outputs for conv2d, fully_connected and max_pool2d.
        conv1_1W = var("conv1_1w", net_data['conv1_1'][0], trainable)
        conv1_1b = var("conv1_1b", net_data['conv1_1'][1], trainable)
        conv1_1in = conv(x, conv1_1W, conv1_1b, 3,3,64,1,1)
        x = tf.nn.relu(conv1_1in)
        
        conv1_2W = var("conv1_2w", net_data['conv1_2'][0], trainable)
        conv1_2b = var("conv1_2b", net_data['conv1_2'][1], trainable)
        conv1_2in = conv(x, conv1_2W, conv1_2b, 3,3,64,1,1)
        x = tf.nn.relu(conv1_2in)
        x = tf.nn.max_pool(x, [1,2,2,1],[1,2,2,1],'VALID')
        
        conv2_1W = var("conv2_1w", net_data['conv2_1'][0], trainable)
        conv2_1b = var("conv2_1b", net_data['conv2_1'][1], trainable)
        conv2_1in = conv(x, conv2_1W, conv2_1b, 3,3,128,1,1)
        x = tf.nn.relu(conv2_1in)
        
        conv2_2W = var("conv2_2w", net_data['conv2_2'][0], trainable)
        conv2_2b = var("conv2_2b", net_data['conv2_2'][1], trainable)
        conv2_2in = conv(x, conv2_2W, conv2_2b, 3,3,128,1,1)
        x = tf.nn.relu(conv2_2in)
        x = tf.nn.max_pool(x, [1,2,2,1],[1,2,2,1],'VALID')
        
        conv3_1W = var("conv3_1w", net_data['conv3_1'][0], trainable)
        conv3_1b = var("conv3_1b", net_data['conv3_1'][1], trainable)
        conv3_1in = conv(x, conv3_1W, conv3_1b, 3,3,256,1,1)
        x = tf.nn.relu(conv3_1in)
        
        conv3_2W = var("conv3_2w", net_data['conv3_2'][0], trainable)
        conv3_2b = var("conv3_2b", net_data['conv3_2'][1], trainable)
        conv3_2in = conv(x, conv3_2W, conv3_2b, 3,3,256,1,1)
        x = tf.nn.relu(conv3_2in)
        
        
        conv3_3W = var("conv3_3w", net_data['conv3_3'][0], trainable)
        conv3_3b = var("conv3_3b", net_data['conv3_3'][1], trainable)
        conv3_3in = conv(x, conv3_3W, conv3_3b, 3,3,256,1,1)
        x = tf.nn.relu(conv3_3in)
        x = tf.nn.max_pool(x, [1,2,2,1],[1,2,2,1],'VALID')
        
        conv4_1W = var("conv4_1w", net_data['conv4_1'][0], trainable)
        conv4_1b = var("conv4_1b", net_data['conv4_1'][1], trainable)
        conv4_1in = conv(x, conv4_1W, conv4_1b, 3,3,512,1,1)
        x = tf.nn.relu(conv4_1in)
        
        conv4_2W = var("conv4_2w", net_data['conv4_2'][0], trainable)
        conv4_2b = var("conv4_2b", net_data['conv4_2'][1], trainable)
        conv4_2in = conv(x, conv4_2W, conv4_2b, 3,3,512,1,1)
        x = tf.nn.relu(conv4_2in)
        
        conv4_3W = var("conv4_3w", net_data['conv4_3'][0], trainable)
        conv4_3b = var("conv4_3b", net_data['conv4_3'][1], trainable)
        conv4_3in = conv(x, conv4_3W, conv4_3b, 3,3,512,1,1)
        x = tf.nn.relu(conv4_3in)
        x = tf.nn.max_pool(x, [1,2,2,1],[1,2,2,1],'VALID')
        
        return x


def shared_trunk(x, reuse=False, dropout=1.0):
    with tf.variable_scope('shared_trunk', reuse=reuse):
        conv1w = new_var('conv1w', [1,1,512,512])
        conv1b = new_var('conv1b', [512])
        x = conv(x, conv1w, conv1b, 1,1,512,1,1, 'VALID')
        x = tf.nn.relu(x)
        # x = tf.nn.dropout(x, dropout)

        conv2w = new_var('conv2w', [5,5,512,128])
        conv2b = new_var('conv2b', [128])
        x = conv(x, conv2w, conv2b, 5,5,128,1,1, 'VALID')
        x = tf.nn.relu(x)
        # x = tf.nn.dropout(x, dropout)

        x = tf.reshape(x, [-1,128*10*10])
        fc1w = new_var('fc1w', [128*10*10,512])
        fc1b = new_var('fc1b', [512])
        x = tf.matmul(x, fc1w) + fc1b
        x = tf.nn.relu(x)
        # x = tf.nn.dropout(x, dropout)

    return x

def shared_trunk_resnet(x, reuse=False, dropout=1.0):
    with tf.variable_scope('shared_trunk', reuse=reuse):
        # conv1w = new_var('conv1_weights', [5,5,1024,128])
        # conv1b = new_var('conv1_bias', [128])
        # x = conv(x, conv1w, conv1b, 5,5,128,1,1, 'VALID')
        # x = tf.nn.relu(x)

        x = tf.reshape(x, [-1,128*10*10])
        fc1w = new_var('fc1_weights', [128*10*10,512])
        fc1b = new_var('fc1_bias', [512])
        x = tf.matmul(x, fc1w) + fc1b

    return x

    
def rebuild_share_trunk(x, reuse = False):
    with tf.variable_scope('shared_trunk', reuse=reuse):
        conv1w = new_var('conv1_weights', [1,1,512,512])
        conv1b = new_var('conv1_bias', [512])
        x = conv(x, conv1w, conv1b, 1,1,512,1,1, 'VALID')
        x = tf.nn.relu(x)

        conv2w = new_var('conv2_weights', [5,5,512,128])
        conv2b = new_var('conv2_bias', [128])
        x = conv(x, conv2w, conv2b, 5,5,128,1,1, 'VALID')
        x = tf.nn.relu(x)

        fc1w = new_var('fc1_weights', [128*10*10,512])
        fc1b = new_var('fc1_bias', [512])


    return x, [fc1w, fc1b]

def seg_head(x, reuse, dropout=1.0):
    with tf.variable_scope('segmentation_head', reuse=reuse):
        fcow = new_var('fco_weights', [512,56*56*2])
        fcob = new_var('fco_bias', [56*56*2])
        x = tf.matmul(x, fcow) + fcob
        x = tf.reshape(x, [-1,56,56,2])
        x = tf.image.resize_images(x, [112,112])
    return x
    
def rebuild_seg_head(reuse = False):
    with tf.variable_scope('segmentation_head', reuse=reuse):
        fcow = new_var('fco_weights', [512,56*56*2])
        fcob = new_var('fco_bias', [56*56*2])
    return [fcow, fcob]

def score_head(x, reuse, dropout=1.0):
    with tf.variable_scope('score_head', reuse=reuse):
        fc1w = new_var('fc1_weights', [512,1024])
        fc1b = new_var('fc1_bias', [1024])
        x = tf.nn.dropout(x, dropout)
        x = tf.matmul(x, fc1w) + fc1b
        x = tf.nn.relu(x)

        fcow = new_var('fco_weights', [1024,2])
        fcob = new_var('fco_bias', [2])
        x = tf.nn.dropout(x, dropout)
        x = tf.matmul(x, fcow) + fcob
    return x

def rebuild_score_head(reuse = False):
    with tf.variable_scope('score_head', reuse=reuse):
        fc1w = new_var('fc1_weights', [512,1024])
        fc1b = new_var('fc1_bias', [1024])

        fcow = new_var('fco_weights', [1024,2])
        fcob = new_var('fco_bias', [2])
    return [fc1w, fc1b, fcow, fcob]
    
def rebuild_network(img_ph, model_path, debug = False):
    #build vgg, no change needed
    vgg_out = vgg_network(img_ph)
    conv_out, share_truck_vars = rebuild_share_trunk(vgg_out)
    seg_head_vars = rebuild_seg_head()
    score_head_vars = rebuild_score_head()
    #load pretrained weightss
    model_saver = tf.train.Saver()
    model_saver.restore(sess, model_path)
    
    #make the model fully convolutionary
    conv_out = Fc2Conv(conv_out, *share_truck_vars)
    conv_out = tf.nn.relu(conv_out) #shared trunk
    
    seg_out = Fc2Conv(conv_out, *seg_head_vars)#out is 1x1x6272
    seg_out = tf.reshape(seg_out, [-1,56,56,2])
    seg_out = tf.image.resize_images(seg_out, [224,224])
    
    score_out = Fc2Conv(conv_out, *score_head_vars[:2])
    score_out = tf.nn.relu(score_out)
    score_out = Fc2Conv(score_out, *score_head_vars[2:]) #1*1*1
    print("finish building graph, all operator have been convolized")
    def model_out(image):
        normalized_image = image/255.0 - 0.5
        if debug:
            return sess.run([seg_out, score_out, conv_out,vgg_out], feed_dict = {img_ph:[normalized_image]})
        return sess.run([seg_out, score_out], feed_dict = {img_ph:[normalized_image]})
    return model_out, sess
    
def build_vgg_network(img_ph, sess=None, trainable=True, dropout=1.0, reuse = False):
    '''
    https://arxiv.org/pdf/1603.08695.pdf
    Head-C architecture
    '''
    x = vgg_network(img_ph, trainable=trainable, reuse=reuse)
    x = shared_trunk(x, reuse=reuse, dropout=dropout)
    mask = seg_head(x, reuse=reuse, dropout=dropout)
    score = score_head(x, reuse=reuse, dropout=dropout)
    return mask, score

def build_resnet50_network(img_ph, sess=None, reuse=False, is_training=True, dropout=1.0):
    x = resnet_50_network(img_ph, reuse=reuse, is_training=is_training)
    tmp_vars = set(tf.all_variables())
    if not reuse:
        saver = tf.train.Saver()
        saver.restore(sess, './data/resnet-50/resnet_v1_50.ckpt')
    x = shared_trunk_resnet(x, reuse=reuse, dropout=dropout)
    mask = seg_head(x, reuse=reuse, dropout=dropout)
    score = score_head(x, reuse=reuse, dropout=dropout)
    sess.run(tf.initialize_variables(set(tf.all_variables()) - tmp_vars))
    return mask, score

def build_resnet18_network(img_ph, sess=None, reuse=False, is_training=True, dropout=1.0):
    x = resnet_18_network(img_ph, reuse=reuse, is_training=is_training)
    x = shared_trunk_resnet(x, reuse=reuse, dropout=dropout)
    mask = seg_head(x, reuse=reuse, dropout=dropout)
    score = score_head(x, reuse=reuse, dropout=dropout)
    return mask, score
    
def Fc2Conv(conv_out, FC_W, FC_b):
    input_channel = conv_out.get_shape().as_list()[-1]
    input_w = int(np.sqrt(FC_W.get_shape().as_list()[0]/input_channel))
    input_whc = [input_w, input_w, input_channel]
    # import pdb; pdb.set_trace()
    fConv_W = tf.reshape(FC_W, input_whc + \
                                        [FC_W.get_shape().as_list()[-1]])
    conv_result = tf.nn.conv2d(conv_out, fConv_W, [1,1,1,1], "VALID")
    return tf.nn.bias_add(conv_result, FC_b)


# Outdated... DO NOT USE

# def score_network(x, reuse=False, dropout=1.0):
#     with tf.variable_scope('scorer', reuse=reuse) as sc:
#         x = tf.nn.max_pool(x, [1,2,2,1],[1,2,2,1],'VALID')
#         x = tf.reshape(x, [-1, 512*7*7])
#         fc1w = new_var('fc1w', [512*7*7,512])
#         fc1b = new_var('fc1b',[512])
#         x = tf.matmul(x, fc1w) + fc1b
#         x = tf.nn.relu(x)
#         x = tf.nn.dropout(x, dropout)
        
#         fc2w = new_var('fc2w', [512,1024])
#         fc2b = new_var('fc2b',[1024])
#         x = tf.matmul(x, fc2w) + fc2b
#         x = tf.nn.relu(x)
#         x = tf.nn.dropout(x, dropout)
        
#         fcow = new_var('fcow', [1024,2])
#         fcob = new_var('fcob',[2])
#         x = tf.matmul(x, fcow) + fcob
#     return x


# def seg_network(x, reuse=False, dropout=1.0):
#     with tf.variable_scope('segmentation', reuse=reuse) as sc:
#         conv1w = new_var('conv1w', [1,1,512,512])
#         conv1b = new_var('conv1b', [512])
#         x = conv(x, conv1w, conv1b, 1,1,512,1,1)
#         x = tf.nn.relu(x)
#         x = tf.nn.dropout(x, dropout)
        
#         x = tf.reshape(x, [-1,512*14*14])
        
#         fc1w = new_var('fc1w', [512*14*14,512])
#         fc1b = new_var('fc1b',[512])
#         x = tf.matmul(x, fc1w) + fc1b
#         x = tf.nn.relu(x)
#         x = tf.nn.dropout(x, dropout)
        
#         fcow = new_var('fcow', [512,56*56*2])
#         fcob = new_var('fcob',[56*56*2])
#         x = tf.matmul(x, fcow) + fcob
        
#         x = tf.reshape(x, [-1,56,56,2])
#         x = tf.image.resize_images(x, [224,224])
        
#     return x
