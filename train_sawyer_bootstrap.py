import tensorflow as tf
from tensorflow.contrib.layers import xavier_initializer

import argparse
import numpy as np
import pickle
import os

from models import  build_resnet50_network, build_resnet18_network
from data_loader import inputs_poking, inputs_sawyer_data
slim = tf.contrib.slim


def gen_name(prefix, ratio, pos_max, neg_min, trunk, decay, lr_factor, background):
    return '%s_%s_lambda_%.2f_a_%d_b_%d_decay_%.5f_lr_factor%.2f_background_%s' % (prefix, \
                            trunk, ratio, pos_max, neg_min, decay, lr_factor, background)


def get_lr(timestep, factor = 1):
    # return 1e-3
    if timestep <= 30000:
        return 1e-3 * factor
    elif timestep <= 60000:
        return 5e-4 * factor
    else:
        return 1e-4 * factor


def main():
    parser = argparse.ArgumentParser()

    # parser.add_argument('pretrains_path', type=str, help='path of pretrained model')
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
    parser.add_argument('--pos_max', type=float, default=22)
    parser.add_argument('--neg_min', type=float, default=66)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--num_itr', type=int, default=200000)
    parser.add_argument('--trunk', type=str, choices=['resnet50', 'resnet18'], default='resnet18')
    parser.add_argument('--add_background', action='store_true')

    args = parser.parse_args()


    train_set_old_names = list([args.train_set_path_old + '/' + l for l in os.listdir(args.train_set_path_old)])
    train_set_new_names = list([args.train_set_path_new + '/' + l for l in os.listdir(args.train_set_path_new)])
    train_set_new_pos_names = list(filter(lambda s: 'pos' in s, train_set_new_names))
    train_set_new_neg_names = list(filter(lambda s: 'neg' in s, train_set_new_names))

    