gpu=$1
trunk=$2
pos_max=$3
neg_min=$4
lr_factor=$5
mask_ratio=$6
decay=$7

CUDA_VISIBLE_DEVICES=$gpu python train_sawyer.py $model --weight_decay $decay --mask_ratio $mask_ratio  --trunk $trunk --pos_max $pos_max --neg_min $neg_min --train_set_path_old /media/icm_data/poke_nlc_training_new --val_set_path /media/icm_data/poke_nlc_val_new --train_set_path_new /media/icm_data/sawyer_data_new --tfmodel_path /media/4tb/dian/window_seg/models --tfboard_path /media/4tb/dian/window_seg/boards
