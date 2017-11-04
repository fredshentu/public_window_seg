gpu=$1
trunk=$2
pos_max=$3
neg_min=$4
factor=$5

CUDA_VISIBLE_DEVICES=$gpu python train_sawyer.py --trunk $trunk --pos_max $pos_max --neg_min $neg_min --factor $factor