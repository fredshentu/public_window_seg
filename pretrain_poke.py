'''
Pretrain our segmentation network with poking data

'''


import tensorflow as tf
from tensorflow.contrib.layers import xavier_initializer

import argparse
import numpy as np
import pickle
import os

from models import build_vgg_network, build_resnet50_network, build_resnet18_network


# scales = tf.convert_to_tensor([2**(-1/4),1.0,1.15, 2**(-1), 2**(-0.75), 2**(-0.6), 2**(0.6), 2**(0.75), 2.0])
scales = tf.convert_to_tensor([2**(-0.25),1.0, 2**(0.25), 2**(0.5), 2**0.75, 2**(-0.5), 2**(-0.75), \
                                            2**(-1.0),2**(1.0)])
jetter_length = None


# def ge

def gen_name(prefix, ratio, pos_max, neg_min, trunk, decay):
    return '%s_%s_lambda_%.2f_a_%d_b_%d_decay_%.5f' % (prefix, trunk, ratio, pos_max, neg_min, decay)

def new_var(name, shape, initializer):
    return tf.get_variable(name, shape=shape, initializer=initializer)

# #assume data is in the format of tfrecord. Want to maintain an replay buffer in disk
def read_decode_positive_example(filename_queue, obs_shape):
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
                            maxval=8,
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


def read_decode_negative_example(filename_queue, obs_shape):
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
                            maxval=5,
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

def inputs(filenames, obs_shape, train=True, batch_size=16, num_epochs = None, positive = True):

    with tf.name_scope('input'):
        filename_queue = tf.train.string_input_producer(
                filenames, num_epochs=num_epochs)
        if positive:
            image, mask, score= read_decode_positive_example(filename_queue, obs_shape)
        
        else:
            image, mask, score = read_decode_negative_example(filename_queue, obs_shape)
            
            
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
                                capacity = queue_capacity, enqueue_many = False)
        
        image = tf.image.resize_images(image, [160,160])
        
        downsampled_mask = tf.image.resize_images(mask, [112,112])
        downsampled_mask = (downsampled_mask + 1) / 2
        downsampled_mask = tf.cast(tf.transpose(downsampled_mask, [3,0,1,2])[0], tf.int32)

        return image, downsampled_mask, score



def get_lr(timestep):
    if timestep <= 7500:
        return 1e-3
    elif timestep <= 15000:
        return 5e-4
    else:
        return 1e-4


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--train_set_path', type=str, default='/media/icm_data/poke_nlc_training_new')
    parser.add_argument('--val_set_path', type=str, default='/media/icm_data/poke_nlc_val_new')
    parser.add_argument('--tfmodel_path', type=str, default='/media/4tb/dian/window_seg/models')
    parser.add_argument('--tfboard_path', type=str, default='/media/4tb/dian/window_seg/boards')

    parser.add_argument('--log_freq', type=int, default=50)
    parser.add_argument('--save_freq', type=int, default=1000)
    parser.add_argument('--gpu_ratio', type=float, default=0.99)

    parser.add_argument('--mask_ratio', type=float, default=32.0)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--pos_max', type=int, default=24)
    parser.add_argument('--neg_min', type=int, default=48)
    parser.add_argument('--initlr', type=float, default=1e-3)
    parser.add_argument('--num_itr', type=int, default=100000)
    parser.add_argument('--trunk', type=str, choices=['resnet50', 'resnet18', 'vgg'])

    args = parser.parse_args()

    global jetter_length
    jetter_length = jetter_length = tf.convert_to_tensor([(0, args.pos_max), (args.neg_min, 64), (0, 64), (64, 90)])



    train_set_names = list([args.train_set_path + '/' + l for l in os.listdir(args.train_set_path)])
    val_set_names = list([args.val_set_path + '/' + l for l in os.listdir(args.val_set_path)])
    
    train_pos_img, train_pos_mask, train_pos_score = inputs(train_set_names, [240,240,4], positive=True)
    train_neg_img, train_neg_mask, train_neg_score = inputs(train_set_names, [240,240,4], positive=False)
    train_img = tf.concat(0, [train_pos_img, train_neg_img])
    train_mask = tf.concat(0, [train_pos_mask, train_neg_mask])
    train_score = tf.concat(0, [train_pos_score, train_neg_score])
    learning_rate = tf.placeholder(tf.float32, [])

    val_pos_img, val_pos_mask, val_pos_score = inputs(val_set_names, [240,240,4], positive=True, train=False)
    val_neg_img, val_neg_mask, val_neg_score = inputs(val_set_names, [240,240,4], positive=False, train=False)
    val_img = tf.concat(0, [val_pos_img, val_neg_img])
    val_mask = tf.concat(0, [val_pos_mask, val_neg_mask])
    val_score = tf.concat(0, [val_pos_score, val_neg_score])


    train_set_names = list([args.train_set_path + '/' + l for l in os.listdir(args.train_set_path)])
    val_set_names = list([args.val_set_path + '/' + l for l in os.listdir(args.val_set_path)])


    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_ratio)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

    # Build network
    if args.trunk == 'vgg':
        train_pred_mask, train_pred_score = build_vgg_network(train_img, dropout=0.5, reuse=False)
        val_pred_mask, val_pred_score = build_vgg_network(val_img, dropout=1.0, reuse=True)
        sess.run(tf.initialize_all_variables())
    elif args.trunk == 'resnet50':
        train_pred_mask, train_pred_score = build_resnet50_network(train_img, sess=sess, reuse=False, is_training=False, dropout=0.5)
        val_pred_mask, val_pred_score = build_resnet50_network(val_img, sess=sess, reuse=True, is_training=False, dropout=1.0)
    elif args.trunk == 'resnet18':
        train_pred_mask, train_pred_score = build_resnet18_network(train_img, sess=sess, reuse=False, is_training=True, dropout=0.5)
        val_pred_mask, val_pred_score = build_resnet18_network(val_img, sess=sess, reuse=True, is_training=False, dropout=1.0)
        sess.run(tf.initialize_all_variables()) # Initialize ResNet params

    tmp_vars = set(tf.all_variables())

    # import pdb; pdb.set_trace()

    train_mask_loss = tf.cast(tf.reshape(train_score,[-1,1,1]), tf.float32) * tf.nn.sparse_softmax_cross_entropy_with_logits(train_pred_mask, train_mask)
    train_label_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(train_pred_score, train_score)
    train_mask_loss = tf.reduce_mean(train_mask_loss)
    train_label_loss = tf.reduce_mean(train_label_loss)
    train_decay_loss = tf.reduce_mean(tf.pack([tf.nn.l2_loss(i) for i in tf.all_variables() if 'weights' in i.name]))
    train_total_loss = args.mask_ratio * train_mask_loss + train_label_loss + args.weight_decay * train_decay_loss


    val_label_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(val_pred_score, val_score)
    val_mask_loss = tf.cast(tf.reshape(val_score,[-1,1,1]), tf.float32) * tf.nn.sparse_softmax_cross_entropy_with_logits(val_pred_mask, val_mask)
    val_mask_loss = tf.reduce_mean(val_mask_loss)
    val_label_loss = tf.reduce_mean(val_label_loss)
    val_total_loss = args.mask_ratio * val_mask_loss + val_label_loss

    optimizer = tf.train.MomentumOptimizer(learning_rate, 0.9)
    # optimizer = tf.train.AdamOptimizer(args.initlr)
    grads, variables = zip(*optimizer.compute_gradients(train_total_loss))
    grads, _ = tf.clip_by_global_norm(grads, 40.0)
    train_opt = optimizer.apply_gradients(zip(grads, variables))

    # mask_optimizer = tf.train.AdamOptimizer(args.mask_ratio * args.initlr)
    # train_mask_opt = mask_optimizer.minimize(train_mask_loss + args.weight_decay * train_decay_loss)
    # label_optimizer = tf.train.AdamOptimizer(args.initlr)
    # train_label_opt = label_optimizer.minimize(train_label_loss + args.weight_decay * train_decay_loss)
    # coin_flip = tf.random_uniform([1],minval=0, maxval=1, dtype=tf.float32)
    # opts = tf.convert_to_tensor([])
    # train_opt = tf.select(tf.greater(tf.random_uniform([minval=min_jettering,maxval=max_jettering,
                                                                        # dtype=tf.int32], ))


    # if args.trunk == 'resnet50' or args.trunk == 'resnet18':
    sess.run(tf.initialize_variables(set(tf.all_variables()) - tmp_vars))

    model_name = gen_name('pretrain_sgd', args.mask_ratio, args.pos_max, args.neg_min, args.trunk, args.weight_decay)

    summary_writer = tf.summary.FileWriter(args.tfboard_path +'/'+model_name, graph=tf.get_default_graph())
    model_saver = tf.train.Saver()


    # import pdb; pdb.set_trace()

    loss_summ = []
    loss_summ.append(tf.summary.scalar('train/mask_loss', train_mask_loss))
    loss_summ.append(tf.summary.scalar('train/label_loss', train_label_loss))
    loss_summ.append(tf.summary.scalar('train/weight_decay_loss', train_decay_loss))
    loss_summ.append(tf.summary.scalar('train/total_loss', train_total_loss))
    loss_summ.append(tf.summary.scalar('val/mask_loss', val_mask_loss))
    loss_summ.append(tf.summary.scalar('val/label_loss', val_label_loss))
    loss_summ.append(tf.summary.scalar('val/total_loss', val_total_loss))

    img_summ = []
    val_pos_img_r, val_pos_img_g, val_pos_img_b = tf.unstack((val_pos_img+0.5)*255.0, axis=-1)
    val_pos_img_viz_r = tf.cast(tf.image.resize_images(val_pos_mask, [160,160]), tf.float32) * 255
    val_pos_img_viz = tf.stack([val_pos_img_viz_r, val_pos_img_g, val_pos_img_b], axis=-1)

    # import pdb; pdb.set_trace()
    val_pos_img_pred_r = tf.unstack(tf. tf.nn.softmax(tf.stack(tf.unstack(val_pred_mask, axis=0)[:16], axis=0), dim=-1),axis=-1)[1] * 255
    val_pos_img_pred_viz = tf.stack([val_pos_img_pred_r, val_pos_img_g, val_pos_img_b], axis=-1)


    img_summ.append(tf.image_summary('val_pos_img', val_pos_img_viz, max_images=10))
    img_summ.append(tf.image_summary('val_pos_img_pred', val_pos_img_pred_viz, max_images=10))
    img_summ.append(tf.image_summary('val_neg_img', val_neg_img, max_images=10))

    loss_summ_op = tf.summary.merge(loss_summ + img_summ)
    # sess.run(tf.initialize_all_variables())

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    print ("Finished creating network")


    for timestep in range(args.num_itr):
        # coin_val = sess.run(coin_flip)
        # if coin_val > 0.5:
        #     train_opt = train_mask_opt
        # else:
        #     train_opt = train_label_opt

        if timestep % args.log_freq == 0:
            print ("Start itr {}".format(timestep))

            _, train_summary = sess.run(
                [
                    train_opt, 
                    loss_summ_op,
                ], {
                    learning_rate : get_lr(timestep)
                }
            )

            summary_writer.add_summary(train_summary, timestep)

        else:
            sess.run([train_opt], { learning_rate: get_lr(timestep) })

        if timestep % args.save_freq == 0:
            model_saver.save(sess, args.tfmodel_path + '/' + model_name, global_step=timestep)
            print ("Saved model to %s" % args.tfmodel_path)

    coord.request_stop()
    sess.close()









if __name__ == '__main__':
   main()

