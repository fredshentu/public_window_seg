gpu=$1
model=$2
trunk=$3
pos_max=$4
neg_min=$5
lr_factor=$6
mask_ratio=$7
decay=$8

CUDA_VISIBLE_DEVICES=$gpu python train_sawyer.py $model --weight_decay $decay --mask_ratio $mask_ratio  --trunk $trunk --pos_max $pos_max --neg_min $neg_min --train_set_path_old /x/fred/poke_nlc_training_new --val_set_path /x/fred/poke_nlc_val_new --train_set_path_new /home/fred/sawyer_data_new --tfmodel_path /x/fred/window_seg/models/11_3_morning --tfboard_path /x/fred/window_seg/boards
