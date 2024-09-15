#!/bin/bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 main_CADS_ssl.py \
--arch='pvtv2_b1' \
--data_path='/media/erwen_SSD/nnUNet_preprocessed/data_SSL/' \
--list_path='SSL_data_deeplesion.txt' \
--output_dir='snapshots/CADS/' \
--batch_size_per_gpu=48 \
--optimizer='adamw' \
--use_fp16=False \
--momentum_teacher=0.996 \
--epochs=150 \
--out_dim=60000 \
--warmup_teacher_temp=0.04 \
--teacher_temp=0.07 \
--warmup_teacher_temp_epochs=50 \
--weight_decay=0.04 \
--weight_decay_end=0.4 \
--lr=0.00075 \
--min_lr=2e-06 \
--use_bn_in_head=False \
--clip_grad=0.3 \
--freeze_last_layer=3 \
--norm_last_layer=True

