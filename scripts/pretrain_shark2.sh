gpu=$1
trunk=$2
pos_max=$3
neg_min=$4

<<<<<<< HEAD
CUDA_VISIBLE_DEVICES=$gpu python pretrain_poke.py --trunk $trunk --pos_max $pos_max --neg_min $neg_min --train_set_path /home/fred/poke_nlc_training_new --val_set_path /home/fred/poke_nlc_val_new --tfmodel_path /x/fred/window_seg/models --tfboard_path /x/fred/window_seg/boards
=======
CUDA_VISIBLE_DEVICES=$gpu python pretrain_poke.py --trunk $trunk --pos_max $pos_max --neg_min $neg_min --train_set_path /x/fred/poke_nlc_training_new --val_set_path /x/fred/poke_nlc_val_new --tfmodel_path /x/fred/window_seg/models --tfboard_path /x/fred/window_seg/boards
>>>>>>> 9c6378328ba55b11d5541fbbcf05f162f3c27db2
