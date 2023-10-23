# coding=utf-8
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# /root/quantization/model/Llama-2-7b-hf
# /tmp/llama2/models/7B-finetuned_2_16_16_aym_gs_16_10

torchrun --nproc_per_node=8 --master_port=15001 train.py \
--local_dir "/tmp/llama2/" \
--input_model_filename "/root/quantization/model/Llama-2-7b-hf" \
--output_model_filename "7B-finetuned_mx24bg32_hess_32_32_aym_1_gdata" \
--train_data_local_path "./gen_data/all_gen_all.jsonl" \
--eval_data_local_path "./gen_data/all_gen_all.jsonl" \
--do_train True \
--do_eval True \
--model_max_length 2048 \
--fp16 False \
--bf16 True \
--log_on_each_node False \
--logging_dir /tmp/output/runs/current \
--num_train_epochs 1 \
--per_device_train_batch_size 2 \
--per_device_eval_batch_size 1 \
--gradient_accumulation_steps 1 \
--evaluation_strategy "no" \
--save_strategy "steps" \
--save_steps 1000 \
--report_to "tensorboard" \
--save_total_limit 1 \
--learning_rate 2e-5 \
--weight_decay 0. \
--warmup_ratio 0. \
--lr_scheduler_type "cosine" \
--logging_steps 1 \
--tf32 False \
--gradient_checkpointing True \
--qat True \
--w_bits $1 \
--a_bits $2 \
--kv_bits $3 \
--use_kd True \
--fsdp "full_shard auto_wrap" \
--fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer'
