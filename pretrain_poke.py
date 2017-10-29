'''
Pretrain our segmentation network with poking data

'''


import tensorflow as tf
from tensorflow.contrib.layers import xavier_initializer

import argparse
import numpy as np
import pickle
import os

from models import vgg_network, score_network, seg_network


# scales = tf.convert_to_tensor([2*(-1/4),1.0,1.15, 2**(-1), 2**(-0.75), 2**(-0.6), 2**(0.6), 2**(0.75), 2.0])
scales = tf.convert_to_tensor([2**(-0.25),1.0, 2**(0.25),2**(-1.0),2**(1.0)])
jetter_length = None

# def ge

def gen_name(prefix, ratio, pos_max, neg_min, freeze):
    return '%s_lambda_%.2f_a_%d_b_%d_vgg_%s' % (prefix, ratio, pos_max, neg_min, freeze)

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
                            maxval=3,
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
            queue_capacity = 10000
        else:
            num_thread = 4
            queue_capacity = 2000
        image, mask,score = tf.train.shuffle_batch([image,mask,score], 
                                min_after_dequeue=1000 , \
                                batch_size = batch_size, \
                                num_threads = num_thread,\
                                capacity = queue_capacity, enqueue_many =False)
        
        image = tf.image.resize_images(image, [224,224])
        
        downsampled_mask = tf.image.resize_images(mask, [224,224])
        downsampled_mask = (downsampled_mask + 1) / 2
        downsampled_mask = tf.cast(tf.transpose(downsampled_mask, [3,0,1,2])[0], tf.int32)

        return image, downsampled_mask, score




def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--train_set_path', type=str, default='/media/icm_data/poke_nlc_training_new')
    parser.add_argument('--val_set_path', type=str, default='/media/icm_data/poke_nlc_val_new')
    parser.add_argument('--tfmodel_path', type=str, default='/media/4tb/dian/window_seg/models')
    parser.add_argument('--tfboard_path', type=str, default='/media/4tb/dian/window_seg/boards')

    parser.add_argument('--log_freq', type=int, default=50)
    parser.add_argument('--save_freq', type=int, default=1000)
    parser.add_argument('--gpu_ratio', type=float, default=0.99)

    parser.add_argument('--score_ratio', type=float, default=1./32)
    parser.add_argument('--pos_max', type=int, default=12)
    parser.add_argument('--neg_min', type=int, default=32)
    parser.add_argument('--initlr', type=float, default=1e-3)
    parser.add_argument('--num_itr', type=int, default=50000)
    parser.add_argument('--train_vgg', action='store_true')

    args = parser.parse_args()

    global jetter_length
    jetter_length = jetter_length = tf.convert_to_tensor([(0, args.pos_max), (args.neg_min, 64), (0, 64), (64,90)])



    train_set_names = list([args.train_set_path + '/' + l for l in os.listdir(args.train_set_path)])
    val_set_names = list([args.val_set_path + '/' + l for l in os.listdir(args.val_set_path)])
    
    train_pos_img, train_pos_mask, train_pos_score = inputs(train_set_names, [240,240,4], positive=True)
    train_neg_img, train_neg_mask, train_neg_score = inputs(train_set_names, [240,240,4], positive=False)
    train_img = tf.concat(0, [train_pos_img, train_neg_img])
    train_mask = tf.concat(0, [train_pos_mask, train_neg_mask])
    train_score = tf.concat(0, [train_pos_score, train_neg_score])

    val_pos_img, val_pos_mask, val_pos_score = inputs(val_set_names, [240,240,4], positive=True, train=False)
    val_neg_img, val_neg_mask, val_neg_score = inputs(val_set_names, [240,240,4], positive=False, train=False)
    val_img = tf.concat(0, [val_pos_img, val_neg_img])
    val_mask = tf.concat(0, [val_pos_mask, val_neg_mask])
    val_score = tf.concat(0, [val_pos_score, val_neg_score])


    train_set_names = list([args.train_set_path + '/' + l for l in os.listdir(args.train_set_path)])
    val_set_names = list([args.val_set_path + '/' + l for l in os.listdir(args.val_set_path)])


    # Build network
    train_vgg_out = vgg_network(train_img, reuse=False, trainable=args.train_vgg)
    train_pred_score = score_network(train_vgg_out, reuse=False, dropout=0.5)
    train_pred_mask = seg_network(train_vgg_out, reuse=False, dropout=0.5)
    val_vgg_out = vgg_network(val_img, reuse=True, trainable=args.train_vgg)
    val_pred_score = score_network(val_vgg_out, reuse=True, dropout=1.0)
    val_pred_mask = seg_network(val_vgg_out, reuse=True, dropout=1.0)

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

    optimizer = tf.train.MomentumOptimizer(args.initlr, 0.9)
    # optimizer = tf.train.AdamOptimizer(args.initlr)
    grads, variables = zip(*optimizer.compute_gradients(train_total_loss))
    grads, _ = tf.clip_by_global_norm(grads, 40.0)
    train_opt = optimizer.apply_gradients(zip(grads, variables))

    model_name = gen_name('pretrain_sgd', args.score_ratio, args.pos_max, args.neg_min, args.train_vgg)

    summary_writer = tf.summary.FileWriter(args.tfboard_path +'/'+model_name, graph=tf.get_default_graph())
    model_saver = tf.train.Saver()

    # import pdb; pdb.set_trace()

    loss_summ = []
    loss_summ.append(tf.summary.scalar('train/mask_loss', train_mask_loss))
    loss_summ.append(tf.summary.scalar('train/label_loss', train_label_loss))
    loss_summ.append(tf.summary.scalar('train/total_loss', train_total_loss))
    loss_summ.append(tf.summary.scalar('val/mask_loss', val_mask_loss))
    loss_summ.append(tf.summary.scalar('val/label_loss', val_label_loss))
    loss_summ.append(tf.summary.scalar('val/total_loss', val_total_loss))

    loss_summ_op = tf.summary.merge(loss_summ)

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_ratio)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    sess.run(tf.initialize_all_variables())

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    print ("Finished creating network")


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

