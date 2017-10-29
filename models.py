'''
Utils for VGG netoworks and seg/scoring network
'''


import tensorflow as tf
from tensorflow.contrib.layers import xavier_initializer

import argparse
import numpy as np
import pickle


with open('./vgg16.pkl', 'rb') as f:
        net_data = pickle.load(f, encoding='latin1') 


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
    return tf.reshape(tf.nn.bias_add(conv, biases), [-1]+conv.get_shape().as_list()[1:])

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
        

def score_network(x, reuse=False, dropout=1.0):
    with tf.variable_scope('scorer', reuse=reuse) as sc:
        x = tf.nn.max_pool(x, [1,2,2,1],[1,2,2,1],'VALID')
        x = tf.reshape(x, [-1, 512*7*7])
        fc1w = new_var('fc1w', [512*7*7,512])
        fc1b = new_var('fc1b',[512])
        x = tf.matmul(x, fc1w) + fc1b
        x = tf.nn.relu(x)
        x = tf.nn.dropout(x, dropout)
        
        fc2w = new_var('fc2w', [512,1024])
        fc2b = new_var('fc2b',[1024])
        x = tf.matmul(x, fc2w) + fc2b
        x = tf.nn.relu(x)
        x = tf.nn.dropout(x, dropout)
        
        fcow = new_var('fcow', [1024,2])
        fcob = new_var('fcob',[2])
        x = tf.matmul(x, fcow) + fcob
    return x


def seg_network(x, reuse=False, dropout=1.0):
    with tf.variable_scope('segmentation', reuse=reuse) as sc:
        conv1w = new_var('conv1w', [1,1,512,512])
        conv1b = new_var('conv1b', [512])
        x = conv(x, conv1w, conv1b, 1,1,512,1,1)
        x = tf.nn.relu(x)
        x = tf.nn.dropout(x, dropout)
        
        x = tf.reshape(x, [-1,512*14*14])
        
        fc1w = new_var('fc1w', [512*14*14,512])
        fc1b = new_var('fc1b',[512])
        x = tf.matmul(x, fc1w) + fc1b
        x = tf.nn.relu(x)
        x = tf.nn.dropout(x, dropout)
        
        fcow = new_var('fcow', [512,56*56*2])
        fcob = new_var('fcob',[56*56*2])
        x = tf.matmul(x, fcow) + fcob
        
        x = tf.reshape(x, [-1,56,56,2])
        x = tf.image.resize_images(x, [224,224])
        
    return x