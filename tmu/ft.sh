#!/bin/bash

n_nodes=1
bs=16
MODEL_PATH=Efficient-Large-Model/VILA1.5-3b
num_train_epochs=1
gradient_accumulation_steps=2
save_steps=100
data_mixture=ai2d_train_12k

num_gpus_per_node=4
gpus=1,2,3,4

PROJECT_NAME='test_pj'
EXP_NAME='test_new'


WANDB_PROJECT=$PROJECT_NAME CUDA_VISIBLE_DEVICES=$gpus torchrun --nnodes=$n_nodes --nproc_per_node=$num_gpus_per_node --master_port=25001 \
    llava/train/train_mem.py \
    --deepspeed ./scripts/zero3.json \
    --model_name_or_path $MODEL_PATH \
    --version v1 \
    --data_mixture $data_mixture\
    --vision_tower google/siglip-so400m-patch14-384 \
    --mm_vision_select_feature cls_patch \
    --mm_projector mlp_downsample \
    --tune_vision_tower False \
    --tune_mm_projector True \
    --tune_language_model True \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio resize \
    --bf16 True \
    --output_dir ./checkpoints/$EXP_NAME \
    --num_train_epochs $num_train_epochs \
    --per_device_train_batch_size $bs \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps $gradient_accumulation_steps \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps $save_steps \
    --save_total_limit 1 \
    --learning_rate 1e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 4096 \
    --gradient_checkpointing True \
    --dataloader_num_workers 16 \
    --lazy_preprocess True \
    --vflan_no_system_prompt True \
    --report_to wandb