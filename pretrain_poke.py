'''
Pretrain our segmentation network with poking data

'''


import tensorflow as tf
from tensorflow.contrib.layers import xavier_initializer

import argparse
import numpy as np
import pickle
import os

from models import build_resnet50_network, build_resnet18_network
from data_loader import inputs_poking
slim = tf.contrib.slim

def gen_name(prefix, ratio, pos_max, neg_min, trunk, background, decay):
    return '%s_%s_lambda_%.2f_a_%d_b_%d_background_%s_decay_%.5f' % (prefix, trunk, ratio, pos_max, neg_min, background, decay)

def new_var(name, shape, initializer):
    return tf.get_variable(name, shape=shape, initializer=initializer)


#resnet-18
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
    parser.add_argument('--tfmodel_path', type=str, default='/home/fred/Desktop/resnet_seg/models')
    parser.add_argument('--tfboard_path', type=str, default='/home/fred/Desktop/resnet_seg/boards')

    parser.add_argument('--log_freq', type=int, default=50)
    parser.add_argument('--save_freq', type=int, default=1000)
    parser.add_argument('--gpu_ratio', type=float, default=0.99)

    parser.add_argument('--mask_ratio', type=float, default=16.0)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--pos_max', type=int, default=24)
    parser.add_argument('--neg_min', type=int, default=46)
    parser.add_argument('--initlr', type=float, default=1e-3)
    parser.add_argument('--num_itr', type=int, default=100000)
    parser.add_argument('--trunk', type=str, choices=['resnet50', 'resnet18', 'vgg'])
    parser.add_argument('--add_background', action='store_true')

    args = parser.parse_args()

   


    train_set_names = list([args.train_set_path + '/' + l for l in os.listdir(args.train_set_path)])
    val_set_names = list([args.val_set_path + '/' + l for l in os.listdir(args.val_set_path)])
    
    train_pos_img, train_pos_mask, train_pos_score, train_pos_background = inputs_poking(train_set_names,\
                                        args.pos_max, args.neg_min, positive=True)
    train_neg_img, train_neg_mask, train_neg_score, train_neg_background = inputs_poking(train_set_names,\
                                        args.pos_max, args.neg_min, positive=False)
    train_img = tf.concat(0, [train_pos_img, train_neg_img])
    train_mask = tf.concat(0, [train_pos_mask, train_neg_mask])
    train_score = tf.concat(0, [train_pos_score, train_neg_score])
    train_background = tf.concat(0, [train_pos_background, train_neg_background])

    learning_rate = tf.placeholder(tf.float32, [])

    val_pos_img, val_pos_mask, val_pos_score, val_pos_background = inputs_poking(val_set_names,args.pos_max, \
                                                    args.neg_min, positive=True, train=False)
    val_neg_img, val_neg_mask, val_neg_score, val_neg_background = inputs_poking(val_set_names, args.pos_max, \
                                                    args.neg_min, positive=False, train=False)
    val_img = tf.concat(0, [val_pos_img, val_neg_img])
    val_mask = tf.concat(0, [val_pos_mask, val_neg_mask])
    val_score = tf.concat(0, [val_pos_score, val_neg_score])
    val_background = tf.concat(0, [val_pos_background, val_neg_background])


    train_set_names = list([args.train_set_path + '/' + l for l in os.listdir(args.train_set_path)])
    val_set_names = list([args.val_set_path + '/' + l for l in os.listdir(args.val_set_path)])


    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_ratio)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

    # Build network
    if args.trunk == 'vgg':
        train_pred_mask, train_pred_score = build_vgg_network(train_img, background=train_background, dropout=0.5, reuse=False, add_background=args.add_background)
        val_pred_mask, val_pred_score = build_vgg_network(val_img, background=val_background, dropout=1.0, reuse=True, add_background=args.add_background)
        sess.run(tf.initialize_all_variables())
    elif args.trunk == 'resnet50':
        train_pred_mask, train_pred_score = build_resnet50_network(train_img, background=train_background, sess=sess, reuse=False, is_training=False, dropout=0.5, add_background=args.add_background)
        val_pred_mask, val_pred_score = build_resnet50_network(val_img, background=val_background, sess=sess, reuse=True, is_training=False, dropout=1.0, add_background=args.add_background)
    elif args.trunk == 'resnet18':
        train_pred_mask, train_pred_score = build_resnet18_network(train_img, background=train_background, sess=sess, reuse=False, is_training=True, dropout=0.5, add_background=args.add_background)
        val_pred_mask, val_pred_score = build_resnet18_network(val_img, background=val_background, sess=sess, reuse=True, is_training=False, dropout=1.0, add_background=args.add_background)
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
    train_opt = slim.learning.create_train_op(train_total_loss, optimizer, clip_gradient_norm=40.0)
    # optimizer = tf.train.AdamOptimizer(args.initlr)
    # grads, variables = zip(*optimizer.compute_gradients(train_total_loss))
    # grads, _ = tf.clip_by_global_norm(grads, 40.0)
    # train_opt = optimizer.apply_gradients(zip(grads, variables))

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

    model_name = gen_name('pretrain_sgd', args.mask_ratio, args.pos_max, args.neg_min, args.trunk, args.background, args.weight_decay)

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
    val_pos_img_r, val_pos_img_g, val_pos_img_b = tf.unstack(tf.image.resize_images((val_pos_img+0.5)*255.0, [112, 112]), axis=-1)
    # import pdb; pdb.set_trace()
    val_pos_img_viz_r = tf.cast(val_pos_mask, tf.float32) * 255
    val_pos_img_viz = tf.stack([val_pos_img_viz_r, val_pos_img_g, val_pos_img_b], axis=-1)

    # import pdb; pdb.set_trace()
    val_pos_img_pred_r = tf.unstack(tf.nn.softmax(tf.stack(tf.unstack(val_pred_mask, axis=0)[:16], axis=0), dim=-1),axis=-1)[1] * 255
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

