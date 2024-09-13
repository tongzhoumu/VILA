#!/bin/zsh

dataset_tag=restroom_v1_2
# eval_data_types=("train" "val_Do_nothing" "val_Move_to_the_left" "val_Move_to_the_right")
ckpt_name=v1_2_3b_clr_3e-6_bs80_ep_50_tune_all
eval_data_type="train"
# eval_data_type="val_Do_nothing"
# eval_data_type="val_Move_to_the_left"
# eval_data_type="val_Move_to_the_right"

for i in {1..10}
do
    echo "#######################################"
    echo "# $i"
    echo "#######################################"
    CUDA_VISIBLE_DEVICES=7 python tmu/evaluate_vqa.py \
        --conv-mode vicuna_v1 \
        --model-path checkpoints/"$ckpt_name"/checkpoint-"$i"0 \
        --image-folder playground/data/onex_"$dataset_tag"_images \
        --annotation-file /mnt/nvme_bulk/home/tmu/data/retrial/"$dataset_tag"_"$eval_data_type".jsonl
done
