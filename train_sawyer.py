import tensorflow as tf
from tensorflow.contrib.layers import xavier_initializer

import argparse
import numpy as np
import pickle
import os

from models import  build_resnet50_network, build_resnet18_network
from data_loader import inputs_poking, inputs_sawyer_data
slim = tf.contrib.slim


def gen_name(prefix, ratio, pos_max, neg_min, trunk, decay, lr_factor):
    return '%s_%s_lambda_%.2f_a_%d_b_%d_decay_%.5f_lr_factor%.2f' % (prefix, \
                            trunk, ratio, pos_max, neg_min, decay, lr_factor)


def get_lr(timestep, factor = 1):
    return 1e-3
    # if timestep <= 30000:
    #     return 1e-3 * factor
    # elif timestep <= 60000:
    #     return 5e-4 * factor
    # else:
    #     return 1e-4 * factor


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('pretrain_path', type=str, help='path of pretrained model')
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
    parser.add_argument('--mask_ratio', type=float, default=32)
    parser.add_argument('--pos_max', type=int, default=22)
    parser.add_argument('--neg_min', type=int, default=66)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--num_itr', type=int, default=200000)
    parser.add_argument('--trunk', type=str, choices=['resnet50', 'resnet18'], default='resnet18')
    parser.add_argument('--add_background', action='store_true')

    args = parser.parse_args()


    train_set_old_names = list([args.train_set_path_old + '/' + l for l in os.listdir(args.train_set_path_old)])
    train_set_new_names = list([args.train_set_path_new + '/' + l for l in os.listdir(args.train_set_path_new)])
    train_set_new_pos_names = list(filter(lambda s: 'pos' in s, train_set_new_names))
    train_set_new_neg_names = list(filter(lambda s: 'neg' in s, train_set_new_names))

    val_set_old_names = list([args.val_set_path + '/' + l for l in os.listdir(args.val_set_path)])

    train_pos_img_poking, train_pos_mask_poking, train_pos_score_poking, train_pos_background_poking = inputs_poking(train_set_old_names,\
                                    args.pos_max, args.neg_min, batch_size = 4, positive=True)
    train_neg_img_poking, train_neg_mask_poking, train_neg_score_poking, train_neg_background_poking = \
                    inputs_poking(train_set_old_names,\
                                        args.pos_max, args.neg_min,  batch_size = 4, positive=False)
    
    
    train_pos_imgs_sawyer, train_pos_masks_sawyer, train_pos_scores_sawyer, train_pos_background = \
                                    inputs_sawyer_data(train_set_new_pos_names, "positive", args.pos_max,\
                                                                    args.neg_min, batch_size = 12)
    train_neg_imgs_sawyer, train_neg_masks_sawyer, train_neg_scores_sawyer, train_neg_background = \
                                    inputs_sawyer_data(train_set_new_neg_names, "negative", args.pos_max, \
                                                                    args.neg_min,batch_size = 6)
    train_neg_from_pos_imgs, train_neg_from_pos_masks, train_neg_from_pos_scores, train_neg_from_pos_background = \
                                    inputs_sawyer_data(train_set_new_pos_names,"negative_from_positive", \
                                                                    args.pos_max, args.neg_min, batch_size = 6)
    
    
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

    train_background = tf.concat(0, [
                                     train_pos_background_poking,
                                     train_neg_background_poking,
                                     train_pos_background,
                                     train_neg_background,
                                     train_neg_from_pos_background,
                                    ])

    
    val_pos_img, val_pos_mask, val_pos_score, val_pos_background = inputs_poking(val_set_old_names,args.pos_max, \
                                                    args.neg_min, positive=True, train=False)
    val_neg_img, val_neg_mask, val_neg_score, val_neg_background = inputs_poking(val_set_old_names, args.pos_max, \
                                                    args.neg_min, positive=False, train=False)
    val_img = tf.concat(0, [val_pos_img, val_neg_img])
    val_mask = tf.concat(0, [val_pos_mask, val_neg_mask])
    val_score = tf.concat(0, [val_pos_score, val_neg_score])
    learning_rate = tf.placeholder(tf.float32, [])
    # Build network
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_ratio)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

    # Build network
    if args.trunk == 'vgg':
        raise NotImplemetedError
    elif args.trunk == 'resnet50':
        train_pred_mask, train_pred_score = build_resnet50_network(train_img, sess=sess, reuse=False, is_training=False, dropout=0.5)
        val_pred_mask, val_pred_score = build_resnet50_network(val_img, sess=sess, reuse=True, is_training=False, dropout=1.0)
        model_saver = tf.train.Saver()
        model_saver.restore(sess, args.pretrain_path)
    elif args.trunk == 'resnet18':
        train_pred_mask, train_pred_score = build_resnet18_network(train_img, background=train_background, sess=sess, reuse=False, is_training=True, dropout=0.5, add_background=args.add_background)
        val_pred_mask, val_pred_score = build_resnet18_network(val_img, background=val_background, sess=sess, reuse=True, is_training=False, dropout=1.0, add_background=args.add_background)
        model_saver = tf.train.Saver()
        model_saver.restore(sess, args.pretrain_path)

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
    
    sess.run(tf.initialize_variables(set(tf.all_variables()) - tmp_vars))

    model_name = gen_name('train_sgd', args.mask_ratio, args.pos_max, args.neg_min, \
                                args.trunk, args.weight_decay, args.lr_factor)

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
            sess.run([train_opt], { learning_rate: get_lr(timestep) })

        if timestep % args.save_freq == 0:
            model_saver.save(sess, args.tfmodel_path + '/' + model_name, global_step=timestep)
            print ("Saved model to %s" % args.tfmodel_path)

    coord.request_stop()
    sess.close()









if __name__ == '__main__':
   main()