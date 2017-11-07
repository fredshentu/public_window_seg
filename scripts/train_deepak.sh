gpu=$1
trunk=$2
pos_max=$3
neg_min=$4
lr_factor=$5
mask_ratio=$6
runid=$7

CUDA_VISIBLE_DEVICES=$gpu python train_sawyer.py --mask_ratio $mask_ratio \
--trunk $trunk --pos_max $pos_max --neg_min $neg_min --lr_factor $lr_factor \
--train_set_path_old /home/fred/poke_nlc_training_new \
--val_set_path /home/fred/poke_nlc_val_new \
--train_set_path_new /home/fred/sawyer_data_new \
--tfmodel_path /home/fred/window_seg/models \
--tfboard_path /home/fred/window_seg/boards \
--no_scale_pos_scoring \
--runid $runid
