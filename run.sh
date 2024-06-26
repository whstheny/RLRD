#!/bin/bash
PROJECT_DIR=/mnt/nfs-storage-pvc-n28/user_codes/rizeJin/wxt/RankCSE-master
SEED=61507
CUDA_VISIBLE_DEVICES=0 \
python train_1.py \
    --simf  \
    --loss_type \
    --corpus_vecs \
    --model_name_or_path \
    --train_file  \
    --output_dir \
    --num_train_epochs \
    --per_device_train_batch_size \
    --learning_rate \
    --max_seq_length\
    --evaluation_strategy steps \
    --load_best_model_at_end \
    --metric_for_best_model  \
    --eval_step \
    --save_steps \
    --pooler_type \
    --mlp_only_train \
    --overwrite_output_dir \
    --temp  \
    --do_train \
    --fp16 \
    --first_teacher_name_or_path \
    --second_teacher_name_or_path \
    --distillation_loss  \
    --alpha_ \
    --beta_ \
    --gamma_  \
    --tau2 \
    --soft_negative_file 