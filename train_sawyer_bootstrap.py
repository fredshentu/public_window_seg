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

from models import  build_resnet50_network, build_resnet18_network
from data_loader import inputs_poking, inputs_sawyer_data
slim = tf.contrib.slim


def main():
    parser = argparse.ArgumentParser()
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
                                        pos_max=args.pos_max, neg_min=args.neg_min, positive=True, addBg=args.add_background, batch_size=8, do_scale_pos_scoring=not args.no_scale_pos_scoring, multi_heads = True)

    train_pos_scoring_img_poke, train_pos_scoring_mask_poke, train_pos_scoring_score_poke, train_pos_scoring_background_poke, train_pos_scoring_head_indicator_poke = inputs_poking(train_set_old_names, \
                                        pos_max=0.1, neg_min=args.neg_min, positive=True, addBg=args.add_background, batch_size=4, multi_heads = True)

    train_neg_img_poke, train_neg_mask_poke, train_neg_score_poke, train_neg_background_poke, train_neg_background_poke, train_neg_head_indicator_poke = inputs_poking(train_set_old_names,\
                                        pos_max=args.pos_max, neg_min=args.neg_min, positive=False, addBg=args.add_background, batch_size=4, multi_heads = True)


    train_pos_segment_img_sawyer, train_pos_segment_mask_sawyer, train_pos_segment_score_sawyer, train_pos_segment_background_sawyer, train_pos_head_indicator_sawyer = \
                                    inputs_sawyer_data(train_set_new_pos_names, "positive", args.pos_max, \
                                    args.neg_min, batch_size = 24, addBg=args.add_background, multi_heads = True)
    train_pos_scoring_img_sawyer, train_pos_scoring_mask_sawyer, train_pos_scoring_score_sawyer, train_pos_scoring_background_sawyer, train_pos_scoring_head_indicator_sawyer = \
                                    inputs_sawyer_data(train_set_new_pos_names, "positive", 0.1, \
                                    args.neg_min, batch_size = 12, addBg=args.add_background, do_scale_pos_scoring=not args.no_scale_pos_scoring, multi_heads = True)

    train_neg_img_sawyer, train_neg_mask_sawyer, train_neg_score_sawyer, train_neg_background_sawyer, train_neg_head_indicator_sawyer = \
                                    inputs_sawyer_data(train_set_new_neg_names, "negative", args.pos_max, \
                                    args.neg_min, batch_size = 6, addBg=args.add_background, multi_heads = True)
    train_neg_from_pos_img_sawyer, train_neg_from_pos_mask_sawyer, train_neg_from_pos_score_sawyer, train_neg_from_pos_background_sawyer, train_neg_from_pos_head_indicator = \
                                    inputs_sawyer_data(train_set_new_pos_names,"negative_from_positive", \
                                    args.pos_max, args.neg_min, batch_size = 6, addBg=args.add_background, multi_heads = True)

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
        train_pos_head_indicator_sawyer
        train_pos_head_indicator_poke
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