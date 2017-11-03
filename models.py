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

def resnet_50_network(img_ph, background=None, is_training=False, reuse=False, scope=None, add_background=False):
    with slim.arg_scope(resnet_v1.resnet_arg_scope()):
      _, end_points = resnet_v1.resnet_v1_50(img_ph, background=background, reuse=reuse, is_training=is_training, add_background=add_background)
    x = end_points['resnet_v1_50/block3/unit_5/bottleneck_v1']
    return x


def resnet_18_network(img_ph, background=None, is_training=False, reuse=False, scope=None, add_background=False):
    with slim.arg_scope(resnet_v1.resnet_arg_scope()):
      _, end_points = resnet_v1.resnet_v1_18(img_ph, background=background, reuse=reuse, is_training=is_training, add_background=False)
    x = end_points['resnet_v1_18/block3/unit_1/bottleneck_v1']
    return x

def shared_trunk_resnet(x, reuse=False, dropout=1.0, add_background=False):
    with tf.variable_scope('shared_trunk', reuse=reuse):
        # import pdb; pdb.set_trace()
        if add_background:
            conv1w = new_var('conv1_weights', [1,1,2048,128])
        else:
            conv1w = new_var('conv1_weights', [1,1,1024,128])
        conv1b = new_var('conv1_bias', [128])
        x = conv(x, conv1w, conv1b, 1,1,128,1,1, 'VALID')
        x = tf.nn.relu(x)

        x = tf.reshape(x, [-1,128*10*10])
        fc1w = new_var('fc1_weights', [128*10*10,512])
        fc1b = new_var('fc1_bias', [512])
        x = tf.matmul(x, fc1w) + fc1b

    return x

def rebuild_shared_trunk_resnet(x, reuse=False, dropout=1.0):
    with tf.variable_scope('shared_trunk', reuse=reuse):
        # import pdb; pdb.set_trace()
        conv1w = new_var('conv1_weights', [1,1,1024,128])
        conv1b = new_var('conv1_bias', [128])
        x = conv(x, conv1w, conv1b, 1,1,128,1,1, 'VALID')
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
    
def rebuild_network(img_ph, model_path, model_type, debug = False):
    sess = tf.Session()
    support_model_type = ['resnet18', 'resnet50', 'VGG']
    assert model_type in support_model_type
    rebuild_shared_trunk = rebuild_shared_trunk_resnet
    if model_type == 'resnet18':
        feature_net = resnet_18_network
    elif model_type == 'resnet50':
        feature_net = resnet_50_network
    elif model_type == 'VGG':
        print("VGG has been purged")
        raise NotImplemetedError
    #build feature network, no change needed
    feature_net_out = feature_net(img_ph, is_training = False)
    
    conv_out, share_truck_vars = rebuild_shared_trunk(feature_net_out)
    seg_head_vars = rebuild_seg_head()
    score_head_vars = rebuild_score_head()
    #load pretrained weights
    model_saver = tf.train.Saver()
    model_saver.restore(sess, model_path)
    
    #make the model fully convolutionary
    conv_out = Fc2Conv(conv_out, *share_truck_vars)

    seg_out = Fc2Conv(conv_out, *seg_head_vars)#out is 1x1x6272
    seg_out = tf.reshape(seg_out, [-1,56,56,2])
    seg_out = tf.image.resize_images(seg_out, [112,112])
    msk = tf.nn.softmax(seg_out)
    
    score_out = Fc2Conv(conv_out, *score_head_vars[:2])
    score_out = tf.nn.relu(score_out)
    score_out = Fc2Conv(score_out, *score_head_vars[2:]) #1*1*1
    score = tf.nn.softmax(score_out)
    
    print("finish building graph, all operator have been convolized")
    def model_out(image, batch = False):
        normalized_image = image/255.0 - 0.5
        if debug:
            return sess.run([seg_out, score_out, conv_out,feature_net_out], feed_dict = {img_ph:[normalized_image]})
        if batch:
            data_in = normalized_image
        else:
            data_in = [normalized_image]
        return sess.run([msk, score], feed_dict = {img_ph:data_in})
    return model_out, sess
    
def rebuild_original_network(img_ph, model_path, model_type, debug = False):
    sess = tf.Session()
    support_model_type = ['resnet18', 'resnet50', 'VGG']
    assert model_type in support_model_type
    if model_type == 'resnet18':
        net = build_resnet18_network
    elif model_type == 'resnet50':
        net = build_resnet50_network
    elif model_type == 'VGG':
        print("VGG has been purged")
        raise NotImplemetedError
    #build feature network, no change needed
    seg_out, score_out = net(img_ph, sess = sess, is_training = False)
    model_saver = tf.train.Saver()
    model_saver.restore(sess, model_path)
    
    msk = tf.nn.softmax(seg_out)
    score = tf.nn.softmax(score_out)
    
    print("finish building original graph")
    def model_out(image, batch = False):
        normalized_image = image/255.0 - 0.5
        if debug:
            return sess.run([seg_out, score_out, conv_out,feature_net_out], feed_dict = {img_ph:[normalized_image]})
        if batch:
            data_in = normalized_image
        else:
            data_in = [normalized_image]
        return sess.run([msk, score], feed_dict = {img_ph:data_in})
    return model_out, sess

def build_resnet50_network(img_ph, background=None, sess=None, reuse=False, is_training=True, dropout=1.0, add_background=False):
    x = resnet_50_network(img_ph, reuse=reuse, is_training=is_training)
    if add_background:
        y = resnet_50_network(background, reuse=True, is_training=is_training)
        x = tf.concat([x, y], axis=-1)

    tmp_vars = set(tf.all_variables())
    if not reuse:
        saver = tf.train.Saver()
        saver.restore(sess, './data/resnet-50/resnet_v1_50.ckpt')
    x = shared_trunk_resnet(x, reuse=reuse, dropout=dropout, add_background=add_background)
    mask = seg_head(x, reuse=reuse, dropout=dropout)
    score = score_head(x, reuse=reuse, dropout=dropout)
    sess.run(tf.initialize_variables(set(tf.all_variables()) - tmp_vars))
    return mask, score

def build_resnet18_network(img_ph, background=None, sess=None, reuse=False, is_training=True, dropout=1.0, add_background=False):
    x = resnet_18_network(img_ph, background=background, reuse=reuse, is_training=is_training)
    if add_background:
        y = resnet_50_network(background, reuse=True, is_training=is_training)
        x = tf.concat([x, y], axis=-1)
    x = shared_trunk_resnet(x, reuse=reuse, dropout=dropout, add_background=add_background)
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