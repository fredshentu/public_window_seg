"""
Bootstrapping deepmask code
                            /->Share trunk1-heads
resnet_18(with or without bk)->Share trunk2-heads
                            \->Share trunk3-heads
data sets are separated
    dataset1
    dataset2
    ...

Current support feature layer architecture: Resnet18

"""

import tensorflow as tf
from tensorflow.contrib.layers import xavier_initializer

import argparse
import numpy as np
import pickle
import os

from models import  build_resnet50_network, build_resnet18_network, build_resnet18_network_bootstrap
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
    parser.add_argument('--runid', type=str, default='')
    parser.add_argument('--train_set_path_old', type=str, default='/media/icm_data/poke_nlc_training_new')
    parser.add_argument('--val_set_path', type=str, default='/media/icm_data/poke_nlc_val_new')
    parser.add_argument('--tfmodel_path', type=str, default='/home/fred/Desktop/window_seg/models')
    parser.add_argument('--tfboard_path', type=str, default='/home/fred/Desktop/window_seg/boards')

    parser.add_argument('--train_set_path_new', type=str, default='/media/icm_data/sawyer_data_mult',
                        help="path of training dataset")

    parser.add_argument('--log_freq', type=int, default=50)
    parser.add_argument('--save_freq', type=int, default=1000)
    parser.add_argument('--gpu_ratio', type=float, default=0.99)
    parser.add_argument('--lr_factor', type=float, default=1)
    parser.add_argument('--mask_ratio', type=float, default=32.0)
    parser.add_argument('--pos_max', type=float, default=24.0)
    parser.add_argument('--neg_min', type=float, default=64.0)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--num_itr', type=int, default=200000)
    parser.add_argument('--num_heads', type = int, default = 5)
    parser.add_argument('--trunk', type=str, choices=['resnet50', 'resnet18'], default='resnet18')
    parser.add_argument('--add_background', action='store_true')
    parser.add_argument('--background_diff_w', action='store_true')
    parser.add_argument('--no_scale_pos_scoring', action='store_true')
    

    args = parser.parse_args()

    train_set_old_names = list([args.train_set_path_old + '/' + l for l in os.listdir(args.train_set_path_old)])
    train_set_new_names = list([args.train_set_path_new + '/' + l for l in os.listdir(args.train_set_path_new)])
    train_set_new_pos_names = list(filter(lambda s: 'pos' in s, train_set_new_names))
    train_set_new_neg_names = list(filter(lambda s: 'neg' in s, train_set_new_names))
    
    train_set_old_names = list([args.train_set_path_old + '/' + l for l in os.listdir(args.train_set_path_old)])
    train_set_new_names = list([args.train_set_path_new + '/' + l for l in os.listdir(args.train_set_path_new)])
    train_set_new_pos_names = list(filter(lambda s: 'pos' in s, train_set_new_names))
    train_set_new_neg_names = list(filter(lambda s: 'neg' in s, train_set_new_names))

    val_set_old_names = list([args.val_set_path + '/' + l for l in os.listdir(args.val_set_path)])
    
    
    #TODO: Update data loader to load head indicator
    train_pos_segment_img_poke, train_pos_segment_mask_poke, train_pos_segment_score_poke, train_pos_segment_background_poke, train_pos_head_indicator_poke = inputs_poking(train_set_old_names,\
                                        pos_max=args.pos_max, neg_min=args.neg_min, positive=True, addBg=args.add_background, batch_size=8, do_scale_pos_scoring=not args.no_scale_pos_scoring, bootstrap = True, num_heads = args.num_heads)

    train_pos_scoring_img_poke, train_pos_scoring_mask_poke, train_pos_scoring_score_poke, train_pos_scoring_background_poke, train_pos_scoring_head_indicator_poke = inputs_poking(train_set_old_names, \
                                        pos_max=0.1, neg_min=args.neg_min, positive=True, addBg=args.add_background, batch_size=4, bootstrap = True, num_heads = args.num_heads)

    train_neg_img_poke, train_neg_mask_poke, train_neg_score_poke, train_neg_background_poke, train_neg_head_indicator_poke = inputs_poking(train_set_old_names,\
                                        pos_max=args.pos_max, neg_min=args.neg_min, positive=False, addBg=args.add_background, batch_size=4, bootstrap = True, num_heads = args.num_heads)


    train_pos_segment_img_sawyer, train_pos_segment_mask_sawyer, train_pos_segment_score_sawyer, train_pos_segment_background_sawyer, train_pos_head_indicator_sawyer = \
                                    inputs_sawyer_data(train_set_new_pos_names, "positive", args.pos_max, \
                                    args.neg_min, batch_size = 44, addBg=args.add_background, bootstrap = True, num_heads = args.num_heads)
    train_pos_scoring_img_sawyer, train_pos_scoring_mask_sawyer, train_pos_scoring_score_sawyer, train_pos_scoring_background_sawyer, train_pos_scoring_head_indicator_sawyer = \
                                    inputs_sawyer_data(train_set_new_pos_names, "positive", 0.1, \
                                    args.neg_min, batch_size = 22, addBg=args.add_background, do_scale_pos_scoring=not args.no_scale_pos_scoring, bootstrap = True, num_heads = args.num_heads)

    train_neg_img_sawyer, train_neg_mask_sawyer, train_neg_score_sawyer, train_neg_background_sawyer, train_neg_head_indicator_sawyer = \
                                    inputs_sawyer_data(train_set_new_neg_names, "negative", args.pos_max, \
                                    args.neg_min, batch_size = 11, addBg=args.add_background, bootstrap = True, num_heads = args.num_heads)
    train_neg_from_pos_img_sawyer, train_neg_from_pos_mask_sawyer, train_neg_from_pos_score_sawyer, train_neg_from_pos_background_sawyer, train_neg_from_pos_head_indicator = \
                                    inputs_sawyer_data(train_set_new_pos_names,"negative_from_positive", \
                                    args.pos_max, args.neg_min, batch_size = 11, addBg=args.add_background, bootstrap = True, num_heads = args.num_heads)

    #segment branch, only positive jettered img
    train_segment_img = tf.concat([
        train_pos_segment_img_sawyer,
        train_pos_segment_img_poke,
    ], 0)

    train_segment_mask = tf.concat([
        train_pos_segment_mask_sawyer,
        train_pos_segment_mask_poke,
    ], 0)

    train_segment_heads_indicator = tf.concat([
        train_pos_head_indicator_sawyer,
        train_pos_head_indicator_poke,
    ], 0)

    #score branch, positive, no jettering no scaling, negative:normal negative
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

    train_score_heads_indicator = tf.concat([
        train_pos_scoring_head_indicator_poke,
        train_pos_scoring_head_indicator_sawyer,
        train_neg_head_indicator_poke,
        train_neg_head_indicator_sawyer,
        train_neg_from_pos_head_indicator,
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
    
    #currently only resnet18
    learning_rate = tf.placeholder(tf.float32, [])
    # Build network
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_ratio)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    _, train_predict_scores = build_resnet18_network_bootstrap(train_scoring_img, background=train_scoring_background,\
                sess=sess, reuse=False, is_training=True, dropout=0.5, add_background=args.add_background, \
                background_diff_w = args.background_diff_w, num_heads = args.num_heads)
    train_predict_masks, _ = build_resnet18_network_bootstrap(train_segment_img, background=train_segment_background,\
                sess=sess, reuse=True, is_training=True, dropout=0.5, add_background=args.add_background, \
                background_diff_w = args.background_diff_w, num_heads = args.num_heads)
    sess.run(tf.initialize_all_variables())
    resnet18_var = tf.trainable_variables(scope='resnet_v1_18')
    background_resnet = tf.trainable_variables(scope='background_resnet')
    all_trainable =  set(tf.trainable_variables())
    bootstrap_share_var = set(resnet18_var + background_resnet)
    tmp_vars = set(tf.all_variables()) # Trunk variables
    unshared_var = all_trainable - bootstrap_share_var
    # unshared_weight_lr_factor = 5.0 * np.ones(len(unshared_var))
    # trainable_variables = list(bootstrap_share_var) + list(unshared_var)
    # shared_vars_names = [var.op.name for var in bootstrap_share_var if "Batch" not in var.op.name]
    # ddebug = [var.op.name for var in bootstrap_share_var]
    # shared_weight_lr_factor = np.ones(len(shared_vars_names))/(1.0*args.num_heads)
    score_branches_unshared = [var.op.name for var in unshared_var if "segmentation_head" not in var.op.name]
    seg_branches_unshared = [var.op.name for var in unshared_var if "score_head" not in var.op.name]
    # import pdb; pdb.set_trace()
    # assert(len(seg_branches_unshared) + len(score_branches_unshared) == len(unshared_var))
    
    score_variables_lr_factor_dict = dict(zip(score_branches_unshared, \
                            np.ones(len(score_branches_unshared))*(args.num_heads)))
    seg_variables_lr_factor_dict = dict(zip(seg_branches_unshared, \
                            np.ones(len(seg_branches_unshared))*(args.num_heads)))
    
    
    train_mask_losses = []
    train_label_losses = []
    train_label_acc = 0
    for i in range(args.num_heads):
        train_pred_mask = train_predict_masks[i]
        train_pred_score = train_predict_scores[i]
        train_mask_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=train_pred_mask,\
                            labels=train_segment_mask) #batchN x 112 x 112
        train_label_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=train_pred_score, \
                            labels=train_scoring_score) #batchN x 1
        train_mask_loss = tf.reduce_mean(train_mask_loss, axis = [1,2])
        # train_label_loss = tf.reduce_mean(train_label_loss, axis = 1)
        train_mask_losses.append(train_mask_loss)
        train_label_losses.append(train_label_loss)
        train_label_acc += tf.reduce_sum(tf.cast(tf.equal(tf.cast(train_scoring_score, tf.int64), \
                            tf.argmax(tf.nn.softmax(train_pred_score, dim=-1), 1)), \
                            tf.float32)* train_score_heads_indicator[:,i])/tf.reduce_sum(train_score_heads_indicator[:,i])
    train_label_acc /= args.num_heads
    
    train_valid_batch_sizes = tf.reduce_sum(tf.transpose(train_segment_heads_indicator), 1)
    #train_mask_losses: batchN * num_heads
    train_mask_losses = tf.reduce_mean(tf.reduce_sum(tf.convert_to_tensor(train_mask_losses)*tf.transpose(train_segment_heads_indicator), 1) / train_valid_batch_sizes)
    #train_mask_labele: batchN * num_heads
    train_label_losses = tf.reduce_mean(tf.reduce_sum(tf.convert_to_tensor(train_label_losses)*tf.transpose(train_segment_heads_indicator), 1) / train_valid_batch_sizes)
    
    decay_loss = tf.reduce_mean(tf.stack([tf.nn.l2_loss(i) for i in tf.all_variables() if 'weights' in i.name]))
    train_total_loss = train_mask_losses * args.mask_ratio + train_label_losses +\
                                    args.weight_decay * decay_loss
    
    mask_optimizer = tf.train.MomentumOptimizer(learning_rate, 0.9)
    score_optimizer = tf.train.MomentumOptimizer(learning_rate, 0.9)
    train_mask_opt = slim.learning.create_train_op(train_mask_losses*args.mask_ratio + args.weight_decay * decay_loss, \
                                                mask_optimizer, clip_gradient_norm=40.0, gradient_multipliers = seg_variables_lr_factor_dict)
    train_score_opt = slim.learning.create_train_op(train_label_losses + args.weight_decay * decay_loss, score_optimizer, \
                                                clip_gradient_norm=40.0, gradient_multipliers = score_variables_lr_factor_dict)
    # import pdb; pdb.set_trace()
    sess.run(tf.initialize_variables(set(tf.all_variables()) - tmp_vars))
    
    if args.runid != '':
        args.runid = args.runid + '_'
    model_name = gen_name('%sbootstrap_heads_%d'%(args.runid, args.num_heads), args.mask_ratio, args.pos_max, args.neg_min, args.trunk, \
                    args.weight_decay, args.lr_factor, args.add_background,args.background_diff_w, \
                    args.no_scale_pos_scoring)

    summary_writer = tf.summary.FileWriter(args.tfboard_path +'/'+model_name, graph=tf.get_default_graph())
    model_saver = tf.train.Saver()

    loss_summ = []
    loss_summ.append(tf.summary.scalar('train/mask_loss', train_mask_losses))
    loss_summ.append(tf.summary.scalar('train/label_loss', train_label_losses))
    loss_summ.append(tf.summary.scalar('train/label_acc', train_label_acc))
    loss_summ.append(tf.summary.scalar('train/total_loss', train_total_loss))
    loss_summ.append(tf.summary.scalar('decay_loss', decay_loss))

    ########img_summ only record the output from the last head###########
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
                    learning_rate : get_lr(timestep, args.lr_factor/(args.num_heads*1.0))
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