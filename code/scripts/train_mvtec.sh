#!/bin/bash

deepspeed --include localhost:0 --master_port 28400 train_mvtec.py \
    --model openllama_peft \
    --stage 1\
    --imagebind_ckpt_path ../pretrained_ckpt/imagebind_ckpt/imagebind_huge.pth\
    --vicuna_ckpt_path ../pretrained_ckpt/vicuna_ckpt/7b_v0/\
    --delta_ckpt_path ../pretrained_ckpt/pandagpt_ckpt/7b/pytorch_model.pt\
    --max_tgt_len 1024\
    --data_path  ../data/small_visual_instruction.json\
    --image_root_path ../data/small_images/\
    --save_path  ./ckpt/train_mvtec/\
    --log_path ./ckpt/train_mvtec/log_rest/

#this is up to date with the google drive version
#notice I am using small_images to test train_mvtech.py
# I also need to create an analogue of data_path for the small dataset. The pandagpt4 thing is for the whole image folder