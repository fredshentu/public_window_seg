import tensorflow as tf
from tensorflow.contrib.layers import xavier_initializer

import argparse
import numpy as np
import pickle
import os

from models import  build_resnet50_network, build_resnet18_network
from data_loader import inputs_poking, inputs_sawyer_data
slim = tf.contrib.slim


def gen_name(prefix, ratio, pos_max, neg_min, trunk, decay, lr_factor, background, bk_share_w,no_scale_pos_scoring):
    return '%s_%s_lambda_%.2f_a_%d_b_%d_decay_%.5f_lr_factor%.2f_background_%s_bk_diff_w_%s_no_scale_pos_%s' % \
                        (prefix, trunk, ratio, pos_max, neg_min, decay, lr_factor, background, bk_share_w, no_scale_pos_scoring)


def get_lr(timestep, factor = 1):
    # return 1e-3
    if timestep <= 50000:
        return 1e-3 * factor
    elif timestep <= 100000:
        return 5e-4 * factor
    else:
        return 1e-4 * factor


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--pretrain_path', type=str, default=None, help='path of pretrained model')
    parser.add_argument('--runid', type=str, default='')
    parser.add_argument('--train_set_path_old', type=str, default='/media/icm_data/poke_nlc_training_new')
    parser.add_argument('--val_set_path', type=str, default='/media/icm_data/poke_nlc_val_new')
    parser.add_argument('--tfmodel_path', type=str, default='/home/fred/Desktop/resnet_seg/models')
    parser.add_argument('--tfboard_path', type=str, default='/home/fred/Desktop/resnet_seg/boards')

    parser.add_argument('--train_set_path_new', type=str, default='/media/icm_data/sawyer_data_new',
                        help="path of training dataset")

    parser.add_argument('--log_freq', type=int, default=50)
    parser.add_argument('--save_freq', type=int, default=1000)
    parser.add_argument('--gpu_ratio', type=float, default=0.99)
    parser.add_argument('--lr_factor', type=float, default=1)
    parser.add_argument('--mask_ratio', type=float, default=24)
    parser.add_argument('--pos_max', type=float, default=22)
    parser.add_argument('--neg_min', type=float, default=66)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--num_itr', type=int, default=200000)
    parser.add_argument('--trunk', type=str, choices=['resnet50', 'resnet18'], default='resnet18')
    parser.add_argument('--add_background', action='store_true')
    parser.add_argument('--background_diff_w', action='store_true')
    parser.add_argument('--no_scale_pos_scoring', action='store_true')
    parser.add_argument('--data_index',type=int, default = 1000)
    
    args = parser.parse_args()

    train_set_old_names = list([args.train_set_path_old + '/' + l for l in os.listdir(args.train_set_path_old)])
    train_set_new_names = list([args.train_set_path_new + '/' + l for l in os.listdir(args.train_set_path_new)])
    train_set_new_pos_names = list(filter(lambda s: 'pos' in s, train_set_new_names))
    train_set_new_neg_names = list(filter(lambda s: 'neg' in s, train_set_new_names))

    # import pdb; pdb.set_trace()
    train_set_new_neg_names.sort(key = lambda a : int(a.split("model")[1].split("_")[0]))
    train_set_new_pos_names.sort(key = lambda a : int(a.split("model")[1].split("_")[0]))
    if len(train_set_new_names) > args.data_index:
        train_set_new_pos_names = train_set_new_pos_names[:args.data_index]
        train_set_new_neg_names = train_set_new_neg_names[:args.data_index]

    val_set_old_names = list([args.val_set_path + '/' + l for l in os.listdir(args.val_set_path)])

    train_pos_segment_img_poke, train_pos_segment_mask_poke, train_pos_segment_score_poke, train_pos_segment_background_poke = inputs_poking(train_set_old_names,\
                                        pos_max=args.pos_max, neg_min=args.neg_min, positive=True, addBg=args.add_background, batch_size=8, do_scale_pos_scoring=not args.no_scale_pos_scoring)

    train_pos_scoring_img_poke, train_pos_scoring_mask_poke, train_pos_scoring_score_poke, train_pos_scoring_background_poke = inputs_poking(train_set_old_names, \
                                        pos_max=0.1, neg_min=args.neg_min, positive=True, addBg=args.add_background, batch_size=4)

    train_neg_img_poke, train_neg_mask_poke, train_neg_score_poke, train_neg_background_poke = inputs_poking(train_set_old_names,\
                                        pos_max=args.pos_max, neg_min=args.neg_min, positive=False, addBg=args.add_background, batch_size=4)


    train_pos_segment_img_sawyer, train_pos_segment_mask_sawyer, train_pos_segment_score_sawyer, train_pos_segment_background_sawyer = \
                                    inputs_sawyer_data(train_set_new_pos_names, "positive", args.pos_max, \
                                                                    args.neg_min, batch_size = 24, addBg=args.add_background)
    train_pos_scoring_img_sawyer, train_pos_scoring_mask_sawyer, train_pos_scoring_score_sawyer, train_pos_scoring_background_sawyer = \
                                    inputs_sawyer_data(train_set_new_pos_names, "positive", 0.1, \
                                                                    args.neg_min, batch_size = 12, addBg=args.add_background, do_scale_pos_scoring=not args.no_scale_pos_scoring)

    train_neg_img_sawyer, train_neg_mask_sawyer, train_neg_score_sawyer, train_neg_background_sawyer = \
                                    inputs_sawyer_data(train_set_new_neg_names, "negative", args.pos_max, \
                                                                    args.neg_min, batch_size = 6, addBg=args.add_background)
    train_neg_from_pos_img_sawyer, train_neg_from_pos_mask_sawyer, train_neg_from_pos_score_sawyer, train_neg_from_pos_background_sawyer = \
                                    inputs_sawyer_data(train_set_new_pos_names,"negative_from_positive", \
                                                                    args.pos_max, args.neg_min, batch_size = 6, addBg=args.add_background)


    train_segment_img = tf.concat([
        train_pos_segment_img_sawyer,
        train_pos_segment_img_poke,
    ], 0)

    train_segment_mask = tf.concat([
        train_pos_segment_mask_sawyer,
        train_pos_segment_mask_poke,
    ], 0)

    train_scoring_img = tf.concat([
        train_pos_scoring_img_poke,
        train_pos_scoring_img_sawyer,
        train_neg_img_poke,
        train_neg_img_sawyer,
        train_neg_from_pos_img_sawyer,
    ], 0)

    train_scoring_score = tf.concat([
        train_pos_scoring_score_poke,
        train_pos_scoring_score_sawyer,
        train_neg_score_poke,
        train_neg_score_sawyer,
        train_neg_from_pos_score_sawyer,
    ], 0)

    train_segment_background = tf.concat([
        train_pos_segment_background_sawyer,
        train_pos_segment_background_poke,
    ], 0)

    train_scoring_background = tf.concat([
        train_pos_scoring_background_poke,
        train_pos_scoring_background_sawyer,
        train_neg_background_poke,
        train_neg_background_sawyer,
        train_neg_from_pos_background_sawyer,
    ], 0)
    # print(train_scoring_background.get_shape().as_list())
    # print(train_segment_background.get_shape().as_list())
    # import pdb; pdb.set_trace()
    


    learning_rate = tf.placeholder(tf.float32, [])
    # Build network
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_ratio)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

    if args.trunk == 'vgg':
        raise NotImplementedError("VGG code is no longer supported --Deepak")
    elif args.trunk == 'resnet50':
        _, train_pred_score = build_resnet50_network(train_scoring_img, background=train_scoring_background,\
                sess=sess, reuse=False, is_training=False, dropout=0.5, add_background=args.add_background)
        train_pred_mask, _  = build_resnet50_network(train_segment_img, background=train_segment_background,\
                sess=sess, reuse=True, is_training=False, dropout=0.5, add_background=args.add_background)
    elif args.trunk == 'resnet18':
        _, train_pred_score = build_resnet18_network(train_scoring_img, background=train_scoring_background,\
                sess=sess, reuse=False, is_training=True, dropout=0.5, add_background=args.add_background, \
                background_diff_w = args.background_diff_w)
        train_pred_mask, _  = build_resnet18_network
        sess.run(tf.initialize_all_variables()) # Initialize ResNet params

    if args.pretrain_path:
        saver = tf.train.Saver()
        saver.restore(sess, args.pretrain_path)


    tmp_vars = set(tf.all_variables()) # Trunk variables

    train_mask_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=train_pred_mask, labels=train_segment_mask)
    train_label_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=train_pred_score, labels=train_scoring_score)
    # train_pred_label_mean = tf.reduce_mean(tf.cast(tf.argmax(tf.nn.softmax(train_pred_score, dim=-1), 1), tf.float32))
    # train_data_label_mean = tf.reduce_mean(tf.cast(train_scoring_score, tf.float32))
    # import pdb; pdb.set_trace()
    train_label_acc = tf.reduce_mean(tf.cast(tf.equal(tf.cast(train_scoring_score, tf.int64), tf.argmax(tf.nn.softmax(train_pred_score, dim=-1), 1)), tf.float32))

    decay_loss = tf.reduce_mean(tf.stack([tf.nn.l2_loss(i) for i in tf.all_variables() if 'weights' in i.name]))
    train_mask_loss = tf.reduce_mean(train_mask_loss)
    train_label_loss = tf.reduce_mean(train_label_loss)
    train_total_loss = train_mask_loss * args.mask_ratio + train_label_loss + args.weight_decay * decay_loss

    mask_optimizer = tf.train.MomentumOptimizer(learning_rate, 0.9)
    score_optimizer = tf.train.MomentumOptimizer(learning_rate, 0.9)
    train_mask_opt = slim.learning.create_train_op(train_mask_loss*args.mask_ratio + args.weight_decay * decay_loss, mask_optimizer, clip_gradient_norm=40.0)
    train_score_opt = slim.learning.create_train_op(train_label_loss + args.weight_decay * decay_loss, score_optimizer, clip_gradient_norm=40.0)

    sess.run(tf.initialize_variables(set(tf.all_variables()) - tmp_vars))

    if args.runid != '':
        args.runid = args.runid + '_'
    model_name = gen_name('%strain_sgd'%args.runid, args.mask_ratio, args.pos_max, args.neg_min, args.trunk, \
                    args.weight_decay, args.lr_factor, args.add_background,args.background_diff_w, \
                    args.no_scale_pos_scoring)

    summary_writer = tf.summary.FileWriter(args.tfboard_path +'/'+model_name, graph=tf.get_default_graph())
    model_saver = tf.train.Saver()

    loss_summ = []
    loss_summ.append(tf.summary.scalar('train/mask_loss', train_mask_loss))
    loss_summ.append(tf.summary.scalar('train/label_loss', train_label_loss))
    loss_summ.append(tf.summary.scalar('train/label_acc', train_label_acc))
    loss_summ.append(tf.summary.scalar('train/total_loss', train_total_loss))
    loss_summ.append(tf.summary.scalar('decay_loss', decay_loss))


    img_summ = []
    train_pos_img_r, train_pos_img_g, train_pos_img_b = tf.unstack(tf.image.resize_images((train_segment_img+0.5)*255.0, [112, 112]), axis=-1)

    train_pos_img_viz_g = (1. - tf.cast(train_segment_mask, tf.float32)) * train_pos_img_g
    train_pos_img_viz_b = (1. - tf.cast(train_segment_mask, tf.float32)) * train_pos_img_b
    train_pos_img_viz = tf.stack([train_pos_img_r, train_pos_img_viz_g, train_pos_img_viz_b], axis=-1)

    train_pos_img_pred_g = tf.unstack(tf.nn.softmax(tf.stack(tf.unstack(train_pred_mask, axis=0), axis=0), dim=-1),axis=-1)[0] * train_pos_img_g
    train_pos_img_pred_b = tf.unstack(tf.nn.softmax(tf.stack(tf.unstack(train_pred_mask, axis=0), axis=0), dim=-1),axis=-1)[0] * train_pos_img_b
    train_pos_img_pred_viz = tf.stack([train_pos_img_r, train_pos_img_pred_g, train_pos_img_pred_b], axis=-1)

    img_summ.append(tf.summary.image('train/pos_img', train_pos_img_viz, max_outputs=10))
    img_summ.append(tf.summary.image('train/pos_img_pred', train_pos_img_pred_viz, max_outputs=10))
    img_summ.append(tf.summary.image('train/neg_img_sawyer', train_neg_img_sawyer, max_outputs=10))
    img_summ.append(tf.summary.image('train/neg_img_poke', train_neg_img_poke, max_outputs=10))

    loss_summ_op = tf.summary.merge(loss_summ + img_summ)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    print ("Finished creating network")

    for timestep in range(args.num_itr):
        if np.random.random() > 0.5:
            train_opt = train_mask_opt
        else:
            train_opt = train_score_opt

        if timestep % args.log_freq == 0:
            print ("Start itr {}".format(timestep))

            _, train_summary = sess.run(
                [
                    train_opt,
                    loss_summ_op,
                ], {
                    learning_rate : get_lr(timestep, args.lr_factor)
                }
            )

            summary_writer.add_summary(train_summary, timestep)

        else:
            sess.run([train_opt], { learning_rate: get_lr(timestep, args.lr_factor) })

        if timestep % args.save_freq == 0:
            model_saver.save(sess, args.tfmodel_path + '/' + model_name, global_step=timestep)
            print ("Saved model to %s" % args.tfmodel_path)

    coord.request_stop()
    sess.close()


if __name__ == '__main__':
   main()
