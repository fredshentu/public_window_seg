import tensorflow as tf
from tensorflow.contrib.layers import xavier_initializer

import argparse
import numpy as np
import pickle
import os

from models import build_vgg_network, build_resnet50_network


scales = tf.convert_to_tensor([2**(-0.25),1.0, 2**(0.25), 2**(0.5), 2**(-0.5), 2**(0.75), 2**(-0.75), 2**(1.), 2**(-1.)])
jetter_length = None

# def ge

def gen_name(prefix, ratio, pos_max, neg_min):
    return '%s_lambda_%.2f_a_%d_b_%d' % (prefix, ratio, pos_max, neg_min)


def read_decode_negative_sawyer_data(filename_queue, obs_shape = [256,256]):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
          serialized_example,
          # Defaults are not specified since both keys are required.
          features={
                  'img': tf.FixedLenFeature([], tf.string),
          })
    img = tf.decode_raw(features['img'], tf.uint8)
    img = tf.reshape(img, [448,448,3])
    img = tf.cast(img, tf.float32)
    img = (img/255.0) - 0.5
    img = tf.pad(img, [[200,200],[200,200],[0,0]])
    random_index = tf.random_uniform(
                            [1],
                            minval=0,
                            maxval=9,
                            dtype=tf.int32)[0]
                            
    crop_size = scales[random_index] * 112

    middle_offset_heigh = 224 + 200 - tf.cast(crop_size, tf.int32)
    middle_offset_width = 224 + 200 - tf.cast(crop_size, tf.int32)
    
    
    
    int_crop_size = tf.cast(scales[random_index] * 224, tf.int32)
    img = tf.image.crop_to_bounding_box(img, middle_offset_heigh, middle_offset_width,\
                                                       int_crop_size , int_crop_size)
    # rotate_k = tf.random_uniform( [1],
    #                         minval=0,
    #                         maxval=4,
    #                         dtype=tf.int32)[0]
    
    # img= tf.image.rot90(img, k=rotate_k, name=None)
    img = tf.image.random_flip_up_down(img)
    img = tf.image.random_flip_left_right(img)
    img = tf.image.resize_images(img, [224,224])
    
    return img,tf.convert_to_tensor(np.zeros([224,224,1])), 0
def read_decode_negative_from_positive_sawyer_data(filename_queue, obs_shape = [256,256]):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
          serialized_example,
          # Defaults are not specified since both keys are required.
          features={
                  'img': tf.FixedLenFeature([], tf.string),
                  'mask':tf.FixedLenFeature([], tf.string),
          })
    img = tf.decode_raw(features['img'], tf.uint8)
    img = tf.reshape(img, [448,448,3])
    img = tf.cast(img, tf.float32)
    img = (img/255.0) - 0.5
    mask = tf.decode_raw(features['mask'],tf.uint8)
    mask = tf.cast(mask, tf.float32)
    mask = tf.reshape(mask, [448,448,1])
    
    img_mask = tf.concat(2,[img, mask])
    img_mask = tf.pad(img_mask, [[200,200],[200,200],[0,0]])
    # rotate_k = tf.random_uniform( [1],
    #                         minval=0,
    #                         maxval=4,
    #                         dtype=tf.int32)[0]
    
    # img_mask= tf.image.rot90(img_mask, k=rotate_k, name=None)
    random_index = tf.random_uniform(
                            [1],
                            minval=0,
                            maxval=9,
                            dtype=tf.int32)[0]
                            
    crop_size = scales[random_index] * 112
    
    jetter_range_index = tf.where(random_index < 3, x = 1, y = 2)
    min_jettering = jetter_length[jetter_range_index][0]
    max_jettering = jetter_length[jetter_range_index][1]
    middle_offset_heigh = 200 + 224 - tf.cast(crop_size, tf.int32)
    middle_offset_width = 200 + 224 - tf.cast(crop_size, tf.int32)
    
    
    elems = tf.cast(tf.convert_to_tensor([1, -1]), tf.int32)
    samples = tf.multinomial(tf.log([[10.,10.]]), 1) # note log-prob
    
    offset_width = middle_offset_width + \
                        elems[tf.cast(samples[0][0], tf.int32)]*tf.random_uniform(
                                                                        [1],
                                                                        minval=min_jettering,
                                                                        maxval=max_jettering,
                                                                        dtype=tf.int32)[0]
    offset_height = middle_offset_heigh + \
                        elems[tf.cast(samples[0][0], tf.int32)]*tf.random_uniform(
                                                                        [1],
                                                                        minval=min_jettering,
                                                                        maxval=max_jettering,
                                                                        dtype=tf.int32)[0]
                                                                        
    int_crop_size = tf.cast(scales[random_index] * 224, tf.int32)
    img_mask = tf.image.crop_to_bounding_box(img_mask, offset_height, offset_width,\
                                                       int_crop_size , int_crop_size)
    
    img_mask = tf.image.random_flip_up_down(img_mask)
    img_mask = tf.image.random_flip_left_right(img_mask)
    image = tf.transpose(tf.gather(tf.transpose(img_mask, [2,0,1]), [0,1,2]), [1,2,0])
    mask = tf.transpose(tf.gather(tf.transpose(img_mask, [2,0,1]), [3]), [1,2,0])
    image = tf.image.resize_images(image, [224,224])
    mask = tf.image.resize_images(mask, [224,224])
    
    return image, mask, 0
    
    
def read_decode_positive_example_sawyer_data(filename_queue, obs_shape = [256,256]):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
          serialized_example,
          # Defaults are not specified since both keys are required.
          features={
                  'img': tf.FixedLenFeature([], tf.string),
                  'mask':tf.FixedLenFeature([], tf.string),
          })
    img = tf.decode_raw(features['img'], tf.uint8)
    img = tf.reshape(img, [448,448,3])
    img = tf.cast(img, tf.float32)
    img = (img/255.0) - 0.5
    mask = tf.decode_raw(features['mask'],tf.uint8)
    mask = tf.cast(mask, tf.float32)
    mask = tf.reshape(mask, [448,448,1])
    
    img_mask = tf.concat(2,[img, mask])
    img_mask = tf.pad(img_mask, [[200,200],[200,200],[0,0]])
    # rotate_k = tf.random_uniform( [1],
    #                         minval=0,
    #                         maxval=4,
    #                         dtype=tf.int32)[0]
    
    # img_mask= tf.image.rot90(img_mask, k=rotate_k, name=None)
    random_index = tf.random_uniform(
                            [1],
                            minval=0,
                            maxval=3,
                            dtype=tf.int32)[0]
    crop_size = scales[random_index] * 112
    
    middle_offset_heigh = 200+224 - tf.cast(crop_size, tf.int32)
    middle_offset_width = 200+224 - tf.cast(crop_size, tf.int32)
    
    
    elems = tf.cast(tf.convert_to_tensor([1, -1]), tf.int32)
    samples = tf.multinomial(tf.log([[10.,10.]]), 1) # note log-prob
    
    offset_width = middle_offset_width + \
                        elems[tf.cast(samples[0][0], tf.int32)]*tf.random_uniform(
                                                                        [1],
                                                                        minval=jetter_length[0][0],
                                                                        maxval=jetter_length[0][1],
                                                                        dtype=tf.int32)[0]
    offset_height = middle_offset_heigh + \
                        elems[tf.cast(samples[0][0], tf.int32)]*tf.random_uniform(
                                                                        [1],
                                                                        minval=jetter_length[0][0],
                                                                        maxval=jetter_length[0][1],
                                                                        dtype=tf.int32)[0]
    int_crop_size = tf.cast(scales[random_index] * 224, tf.int32)
    img_mask = tf.image.crop_to_bounding_box(img_mask, offset_height, offset_width,\
                                                       int_crop_size , int_crop_size)
    
    img_mask = tf.image.random_flip_up_down(img_mask)
    img_mask = tf.image.random_flip_left_right(img_mask)
    image = tf.transpose(tf.gather(tf.transpose(img_mask, [2,0,1]), [0,1,2]), [1,2,0])
    mask = tf.transpose(tf.gather(tf.transpose(img_mask, [2,0,1]), [3]), [1,2,0])
    image = tf.image.resize_images(image, [224,224])
    mask = tf.image.resize_images(mask, [224,224])
    
    return image, mask,1
def inputs_sawyer_data(filenames, mode, train=True, batch_size=12, num_epochs = None):
    assert (mode in set(["positive", "negative", "negative_from_positive"]))
    with tf.name_scope('input'):
        filename_queue = tf.train.string_input_producer(
                filenames, num_epochs=num_epochs, shuffle=True)
        if mode == "positive":
            image, mask, score= read_decode_positive_example_sawyer_data(filename_queue)
        
        elif mode == "negative":
            image, mask, score = read_decode_negative_sawyer_data(filename_queue)
        elif mode == "negative_from_positive":
            image, mask, score = read_decode_negative_from_positive_sawyer_data(filename_queue)
        
        if train:
            num_thread = 20
            queue_capacity = 3000
        else:
            num_thread = 4
            queue_capacity = 3000
        image, mask,score = tf.train.shuffle_batch([image,mask,score], 
                                min_after_dequeue=1000 , \
                                batch_size = batch_size, \
                                num_threads = num_thread,\
                                capacity = queue_capacity, enqueue_many =False)
        
        
        mask = tf.cast(tf.transpose(mask, [3,0,1,2])[0], tf.int32)
        mask = tf.image.resize_images(mask, [112,112])
        image = tf.image.resize_images(image, [160,160])
        
        return image, mask, score



def read_decode_positive_example_poking_data(filename_queue, obs_shape):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
          serialized_example,
          # Defaults are not specified since both keys are required.
          features={
                  'data': tf.FixedLenFeature([np.prod(obs_shape)], tf.float32),
                  'center_max_axis':tf.FixedLenFeature([3], tf.float32),
          })
    center_max_axis = tf.cast(features['center_max_axis'], tf.float32)
    center_max_axis = tf.reshape(center_max_axis, [3])
    data = tf.cast(features['data'], tf.float32)
    data = tf.reshape(data, obs_shape)
    
    center_x = center_max_axis[0] 
    center_y = center_max_axis[1]
    long_axis = center_max_axis[2]
    
    
    #scaling to this data
    random_index = tf.random_uniform(
                            [1],
                            minval=0,
                            maxval=9,
                            dtype=tf.int32)[0]
    scale = scales[random_index] * (224.0/long_axis)
    center_x = center_max_axis[0] * scale + 200
    center_y = center_max_axis[1] * scale + 200
    int_final_size = tf.cast(240*scale, tf.int32)
    data = tf.image.resize_images(data, (int_final_size, int_final_size))
    
    
    
    data = tf.pad(data, [[200,200],[200,200],[0,0]])



    middle_offset_heigh = tf.cast(center_y - 112, tf.int32)
    middle_offset_width = tf.cast(center_x - 112, tf.int32)
    
    
    elems = tf.cast(tf.convert_to_tensor([1, -1]), tf.int32)
    samples = tf.multinomial(tf.log([[10.,10.]]), 1) # note log-prob
    offset_width = middle_offset_width + \
                        elems[tf.cast(samples[0][0], tf.int32)]*tf.random_uniform(
                                                                        [1],
                                                                        minval=jetter_length[0][0],
                                                                        maxval=jetter_length[0][1],
                                                                        dtype=tf.int32)[0]
    offset_height = middle_offset_heigh + \
                        elems[tf.cast(samples[0][0], tf.int32)]*tf.random_uniform(
                                                                        [1],
                                                                        minval=jetter_length[0][0],
                                                                        maxval=jetter_length[0][1],
                                                                        dtype=tf.int32)[0]
    data = tf.image.crop_to_bounding_box(data, offset_height, offset_width,\
                                                       224 , 224)
    data = tf.image.random_flip_up_down(data)
    data = tf.image.random_flip_left_right(data)
    image = tf.transpose(tf.gather(tf.transpose(data, [2,0,1]), [0,1,2]), [1,2,0])
    mask = tf.transpose(tf.gather(tf.transpose(data, [2,0,1]), [3]), [1,2,0])
    
    return image, mask,1


def read_decode_negative_example_poking_data(filename_queue, obs_shape):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
          serialized_example,
          # Defaults are not specified since both keys are required.
          features={
                  'data': tf.FixedLenFeature([np.prod(obs_shape)], tf.float32),
                  'center_max_axis':tf.FixedLenFeature([3], tf.float32),
          })
    center_max_axis = tf.cast(features['center_max_axis'], tf.float32)
    center_max_axis = tf.reshape(center_max_axis, [3])
    data = tf.cast(features['data'], tf.float32)
    data = tf.reshape(data, obs_shape)
    
    center_x = center_max_axis[0] 
    center_y = center_max_axis[1]
    long_axis = center_max_axis[2]
    #scaling to this data
    random_index = tf.random_uniform(
                            [1],
                            minval=0,
                            maxval=9,
                            dtype=tf.int32)[0]
    #if good range, using bad offset, if bad range, using arbitrary offset
    jetter_range_index = tf.where(random_index < 3, x = 1, y = 2)
    #sometimes use large offset to learn about padding and arena
    
    jetter_range_index_final = tf.where(random_index < 2, x = 3, y = jetter_range_index)
    min_jettering = jetter_length[jetter_range_index_final][0]
    max_jettering = jetter_length[jetter_range_index_final][1]
    
    
    
    
    scale = scales[random_index] * (224.0/long_axis)
    center_x = center_max_axis[0] * scale + 200
    center_y = center_max_axis[1] * scale + 200
    int_final_size = tf.cast(240*scale, tf.int32)
    data = tf.image.resize_images(data, (int_final_size, int_final_size))
    
    
    
    data = tf.pad(data, [[200,200],[200,200],[0,0]])



    middle_offset_heigh = tf.cast(center_y - 112, tf.int32)
    middle_offset_width = tf.cast(center_x - 112, tf.int32)
    
    
    elems = tf.cast(tf.convert_to_tensor([1, -1]), tf.int32)
    samples = tf.multinomial(tf.log([[10.,10.]]), 1) # note log-prob
    offset_width = middle_offset_width + \
                        elems[tf.cast(samples[0][0], tf.int32)]*tf.random_uniform(
                                                                        [1],
                                                                        minval=min_jettering,
                                                                        maxval=max_jettering,
                                                                        dtype=tf.int32)[0]
    offset_height = middle_offset_heigh + \
                        elems[tf.cast(samples[0][0], tf.int32)]*tf.random_uniform(
                                                                        [1],
                                                                        minval=min_jettering,
                                                                        maxval=max_jettering,
                                                                        dtype=tf.int32)[0]
    data = tf.image.crop_to_bounding_box(data, offset_height, offset_width,\
                                                       224 , 224)
    data = tf.image.random_flip_up_down(data)
    data = tf.image.random_flip_left_right(data)
    image = tf.transpose(tf.gather(tf.transpose(data, [2,0,1]), [0,1,2]), [1,2,0])
    mask = tf.transpose(tf.gather(tf.transpose(data, [2,0,1]), [3]), [1,2,0])

    return image, mask, 0

def inputs_poking_data(filenames, obs_shape, train=True, batch_size=16, num_epochs = None, positive = True):

    with tf.name_scope('input'):
        filename_queue = tf.train.string_input_producer(
                filenames, num_epochs=num_epochs, shuffle=True)
        if positive:
            image, mask, score = read_decode_positive_example_poking_data(filename_queue, obs_shape)
        
        else:
            image, mask, score = read_decode_negative_example_poking_data(filename_queue, obs_shape)
            
            
        if train:
            num_thread = 12
            queue_capacity = 3000
        else:
            num_thread = 4
            queue_capacity = 3000
        image, mask,score = tf.train.shuffle_batch([image,mask,score], 
                                min_after_dequeue=1000 , \
                                batch_size = batch_size, \
                                num_threads = num_thread,\
                                capacity = queue_capacity, enqueue_many =False)
        
        image = tf.image.resize_images(image, [160,160])
        
        downsampled_mask = tf.image.resize_images(mask, [112,112])
        downsampled_mask = (downsampled_mask + 1) / 2
        downsampled_mask = tf.cast(tf.transpose(downsampled_mask, [3,0,1,2])[0], tf.int32)

        return image, downsampled_mask, score

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('pretrain_path', type=str, help='path of pretrained model')
    parser.add_argument('--train_set_path', type=str, default='/media/icm_data/poke_nlc_training_new')
    parser.add_argument('--val_set_path', type=str, default='/media/icm_data/poke_nlc_val_new')
    parser.add_argument('--tfmodel_path', type=str, default='/media/4tb/dian/window_seg/models')
    parser.add_argument('--tfboard_path', type=str, default='/media/4tb/dian/window_seg/boards')

    parser.add_argument('--train_set_path_old', type=str, default='/media/icm_data/poking_nlc_train',
                        help="path of training dataset")
    parser.add_argument('--train_set_path_new', type=str, default='/media/icm_data/sawyer_data_new',
                        help="path of training dataset")

    parser.add_argument('--log_freq', type=int, default=50)
    parser.add_argument('--save_freq', type=int, default=1000)
    parser.add_argument('--gpu_ratio', type=float, default=0.99)

    parser.add_argument('--score_ratio', type=float, default=1./32)
    parser.add_argument('--pos_max', type=int, default=9)
    parser.add_argument('--neg_min', type=int, default=48)
    parser.add_argument('--initlr', type=float, default=1e-3)
    parser.add_argument('--num_itr', type=int, default=50000)
    parser.add_argument('--trunk', type=str, choices=['vgg', 'resnet50'], default='vgg')

    args = parser.parse_args()

    global jetter_length
    jetter_length = tf.convert_to_tensor([(0, args.pos_max), (args.neg_min, 96), (0, 96), (96,120)])

    train_set_old_names = list([args.train_set_path_old + '/' + l for l in os.listdir(args.train_set_path_old)])
    train_set_new_names = list([args.train_set_path_new + '/' + l for l in os.listdir(args.train_set_path_new)])
    train_set_new_pos_names = list(filter(lambda s: 'pos' in s, train_set_new_names))
    train_set_new_neg_names = list(filter(lambda s: 'neg' in s, train_set_new_names))

    val_set_old_names = list([args.val_set_path + '/' + l for l in os.listdir(args.val_set_path)])

    train_pos_img_poking, train_pos_mask_poking, train_pos_score_poking = \
                    inputs_poking_data(train_set_old_names, [240,240,4], batch_size = 4, positive=True)
    train_neg_img_poking, train_neg_mask_poking, train_neg_score_poking = \
                    inputs_poking_data(train_set_old_names, [240,240,4], batch_size = 4, positive=False)
                    
    train_pos_imgs_sawyer, train_pos_masks_sawyer, train_pos_scores_sawyer = \
                                    inputs_sawyer_data(train_set_new_pos_names, "positive", batch_size = 12)
    train_neg_imgs_sawyer, train_neg_masks_sawyer, train_neg_scores_sawyer = \
                                    inputs_sawyer_data(train_set_new_neg_names, "negative", batch_size = 6)
    train_neg_from_pos_imgs, train_neg_from_pos_masks, train_neg_from_pos_scores = inputs_sawyer_data(train_set_new_pos_names, \
                                                                        "negative_from_positive", batch_size = 6)
                                                                    
    
    train_img = tf.concat(0, [train_pos_img_poking,\
                                train_neg_img_poking, \
                                train_pos_imgs_sawyer, \
                                train_neg_imgs_sawyer, \
                                train_neg_from_pos_imgs])
                                
    train_mask = tf.concat(0, [train_pos_mask_poking, \
                                train_neg_mask_poking,\
                                train_pos_masks_sawyer,\
                                train_neg_masks_sawyer,\
                                train_neg_from_pos_masks])
                                
    train_score = tf.concat(0, [train_pos_score_poking, \
                                train_neg_score_poking, \
                                train_pos_scores_sawyer,\
                                train_neg_scores_sawyer,\
                                train_neg_from_pos_scores])

    val_pos_img, val_pos_mask, val_pos_score = inputs_poking_data(val_set_old_names, [240,240,4], positive=True, train=False)
    val_neg_img, val_neg_mask, val_neg_score = inputs_poking_data(val_set_old_names, [240,240,4], positive=False, train=False)
    val_img = tf.concat(0, [val_pos_img, val_neg_img])
    val_mask = tf.concat(0, [val_pos_mask, val_neg_mask])
    val_score = tf.concat(0, [val_pos_score, val_neg_score])

    # Build network
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_ratio)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

    if args.trunk == 'vgg':
        train_pred_mask, train_pred_score = build_vgg_network(train_img, trainable=True, reuse=False, dropout=0.5)
        val_pred_mask, val_pred_score = build_vgg_network(val_img, trainable=True, reuse=True, dropout=1.0)
    elif args.trunk == 'resnet50':
        train_pred_mask, train_pred_score = build_resnet50_network(train_img, sess, is_training=True, reuse=False, dropout=0.5)
        val_pred_mask, val_pred_score = train_pred_mask, train_pred_score
    

    saver = tf.train.Saver()
    saver.restore(sess, args.pretrain_path)
    print ("Finished loading network")

    train_mask_loss = tf.cast(tf.reshape(train_score,[-1,1,1]), tf.float32) * tf.nn.sparse_softmax_cross_entropy_with_logits(train_pred_mask, train_mask)
    train_label_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(train_pred_score, train_score)
    train_mask_loss = tf.reduce_mean(train_mask_loss)
    train_label_loss = tf.reduce_mean(train_label_loss)
    train_total_loss = train_mask_loss + args.score_ratio * train_label_loss


    val_label_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(val_pred_score, val_score)
    val_mask_loss = tf.cast(tf.reshape(val_score,[-1,1,1]), tf.float32) * tf.nn.sparse_softmax_cross_entropy_with_logits(val_pred_mask, val_mask)
    val_mask_loss = tf.reduce_mean(val_mask_loss)
    val_label_loss = tf.reduce_mean(val_label_loss)
    val_total_loss = val_mask_loss + args.score_ratio * val_label_loss

    tmp_vars = set(tf.all_variables())

    optimizer = tf.train.MomentumOptimizer(args.initlr, 0.9)
    # optimizer = tf.train.AdamOptimizer(args.initlr)
    grads, variables = zip(*optimizer.compute_gradients(train_total_loss))
    grads, _ = tf.clip_by_global_norm(grads, 40.0)
    train_opt = optimizer.apply_gradients(zip(grads, variables))

    # Reinitialize optimizers
    sess.run(tf.initialize_variables(set(tf.all_variables()) - tmp_vars))

    model_name = gen_name('train_sgd', args.score_ratio, args.pos_max, args.neg_min)

    summary_writer = tf.summary.FileWriter(args.tfboard_path +'/'+model_name, graph=tf.get_default_graph())
    model_saver = tf.train.Saver()

    loss_summ = []
    loss_summ.append(tf.summary.scalar('train/mask_loss', train_mask_loss))
    loss_summ.append(tf.summary.scalar('train/label_loss', train_label_loss))
    loss_summ.append(tf.summary.scalar('train/total_loss', train_total_loss))
    loss_summ.append(tf.summary.scalar('val/mask_loss', val_mask_loss))
    loss_summ.append(tf.summary.scalar('val/label_loss', val_label_loss))
    loss_summ.append(tf.summary.scalar('val/total_loss', val_total_loss))

    loss_summ_op = tf.summary.merge(loss_summ)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)


    for timestep in range(args.num_itr):
        if timestep % args.log_freq == 0:
            print ("Start itr {}".format(timestep))

            _, train_summary = sess.run(
                [
                    train_opt, 
                    loss_summ_op,
                ]
            )

            summary_writer.add_summary(train_summary, timestep)

        else:
            sess.run([train_opt])

        if timestep % args.save_freq == 0:
            model_saver.save(sess, args.tfmodel_path + '/' + model_name, global_step=timestep)
            print ("Saved model to %s" % args.tfmodel_path)

    coord.request_stop()
    sess.close()



if __name__ == '__main__':
    main()