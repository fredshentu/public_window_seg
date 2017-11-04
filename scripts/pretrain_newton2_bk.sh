gpu=$1
trunk=$2
pos_max=$3
neg_min=$4

CUDA_VISIBLE_DEVICES=$gpu python pretrain_poke.py --trunk $trunk --pos_max $pos_max --neg_min $neg_min --add_background
