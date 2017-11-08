gpu=$1
trunk=$2
pos_max=$3
neg_min=$4
lr_factor=$5
mask_ratio=$6
bk_share_w=$7
runid=$8

if [ $bk_share_w -eq 1 ]
then
    echo "unshare_weight code launching .."
    CUDA_VISIBLE_DEVICES=$gpu python train_sawyer.py --mask_ratio $mask_ratio \
    --trunk $trunk --pos_max $pos_max --neg_min $neg_min --lr_factor $lr_factor \
    --train_set_path_old /home/fred/poke_nlc_training_new \
    --val_set_path /home/fred/poke_nlc_val_new \
    --train_set_path_new /home/fred/sawyer_data_new \
    --tfmodel_path /home/fred/window_seg/models \
    --tfboard_path /home/fred/window_seg/boards \
    --no_scale_pos_scoring \
    --runid $runid \
    --add_background \
    --background_diff_w
else
    echo "shared_weight code launching .."
    CUDA_VISIBLE_DEVICES=$gpu python train_sawyer.py --mask_ratio $mask_ratio \
    --trunk $trunk --pos_max $pos_max --neg_min $neg_min --lr_factor $lr_factor \
    --train_set_path_old /home/fred/poke_nlc_training_new \
    --val_set_path /home/fred/poke_nlc_val_new \
    --train_set_path_new /home/fred/sawyer_data_new \
    --tfmodel_path /home/fred/window_seg/models \
    --tfboard_path /home/fred/window_seg/boards \
    --no_scale_pos_scoring \
    --runid $runid \
    --add_background
fi
