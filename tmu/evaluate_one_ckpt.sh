#!/bin/zsh

# dataset_tag=restroom_v1_2
# eval_data_types=("train" "val_Do_nothing" "val_Move_to_the_left" "val_Move_to_the_right")
# ckpt_name=v1_2_13b_lr_3e-5_bs80_ep_100_tune_vision

# for eval_data_type in $eval_data_types
# do
#     echo "======== Evaluating on $eval_data_type ========="
#     CUDA_VISIBLE_DEVICES=7 python tmu/evaluate_vqa.py \
#         --conv-mode vicuna_v1 \
#         --model-path checkpoints/$ckpt_name \
#         --image-folder playground/data/onex_"$dataset_tag"_images \
#         --annotation-file /mnt/nvme_bulk/home/tmu/data/retrial/"$dataset_tag"_"$eval_data_type".jsonl
# done


version=v4_0
# eval_data_types=("val_Pick_up_paper" "val_Take_paper_from_machine" "train_Pick_up_paper")
eval_data_types=("val_Pick_up_paper" "val_Take_paper_from_machine")
ckpt_name=sbr_paper_"$version"_3b_lr_1e-5_bs128_ep_40_tune_all

for eval_data_type in $eval_data_types
do
    echo "======== Evaluating on $eval_data_type ========="
    CUDA_VISIBLE_DEVICES=7 python tmu/evaluate_vqa.py \
        --conv-mode vicuna_v1 \
        --model-path checkpoints/$ckpt_name \
        --image-folder playground/data/onex_sbr_paper_"$version"_images \
        --annotation-file /mnt/nvme_bulk/home/tmu/data/sbr/paper_"$version"_"$eval_data_type".jsonl
done
