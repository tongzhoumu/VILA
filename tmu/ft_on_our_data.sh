#!/bin/bash

n_nodes=1
# MODEL_PATH=Efficient-Large-Model/VILA1.5-3b
# version=v1
# MODEL_PATH=Efficient-Large-Model/Llama-3-VILA1.5-8B
# version=llama_3
MODEL_PATH=Efficient-Large-Model/VILA1.5-13b
version=v1
num_train_epochs=2
gradient_accumulation_steps=1
save_steps=100
n_ckpt=1

num_gpus_per_node=8
gpu_ids=0,1,2,3,4,5,6,7
bs=10
gradient_accumulation_steps=1

PROJECT_NAME='VLM'
data_tag=v1_2
data_mixture=onex_restroom_$data_tag

# num_gpus_per_node=2
# gpu_ids=6,7
# bs=40
total_bs=$((num_gpus_per_node*bs))
num_train_epochs=100
# save_strategy="epoch"
save_strategy="steps"
save_steps=1000000000000
n_ckpt=10
lr_scheduler='cosine'
lr=3e-5
# lr_scheduler='constant'
# lr=3e-6
tune_vision=True
# tune_vision=False
# tune_llm=True
tune_llm=False
EXP_NAME="$data_tag"_13b_lr_"$lr"_bs"$total_bs"_ep_"$num_train_epochs"_tune_vision


WANDB_PROJECT=$PROJECT_NAME CUDA_VISIBLE_DEVICES=$gpu_ids torchrun --nnodes=$n_nodes --nproc_per_node=$num_gpus_per_node --master_port=25001 \
    llava/train/train_mem.py \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path $MODEL_PATH \
    --version $version \
    --data_mixture $data_mixture\
    --vision_tower google/siglip-so400m-patch14-384 \
    --mm_vision_select_feature cls_patch \
    --mm_projector mlp_downsample \
    --tune_vision_tower $tune_vision \
    --tune_mm_projector True \
    --tune_language_model $tune_llm \
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
    --save_strategy $save_strategy \
    --save_steps $save_steps \
    --save_total_limit $n_ckpt \
    --learning_rate $lr \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type $lr_scheduler \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 4096 \
    --gradient_checkpointing True \
    --dataloader_num_workers 16 \
    --lazy_preprocess True \
    --vflan_no_system_prompt True \
    --report_to wandb