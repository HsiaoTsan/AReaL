#!/bin/bash

# max_staleness=4
CUDA_VISIBLE_DEVICES=4,5,6,7 python eval_and_aggregate.py \
  --model_path "/data1/large-model-inference/dataset/models_training_dataset/tmp/areal/experiments/checkpoints/root/gsm8k-grpo-stale-4/trial-attn_backend-triton/default/epoch9epochstep28globalstep289" \
  --output_path "/data1/large-model-inference/dataset/models_training_dataset/tmp/areal/experiments/checkpoints/root/gsm8k-grpo-stale-4/trial-attn_backend-triton/default/epoch9epochstep28globalstep289/eval" \
  --max_gen_tokens 32768 \
  --data_names aime24,aime25,amc23,math_500,gpqa_diamond \
  --prompt_type qwen3-think \
  --task math
sleep 60

# max_staleness=3
CUDA_VISIBLE_DEVICES=4,5,6,7 python eval_and_aggregate.py \
  --model_path "/data1/large-model-inference/dataset/models_training_dataset/tmp/areal/experiments/checkpoints/root/gsm8k-grpo-stale-3/trial2-attn_backend-triton/default/epoch9epochstep28globalstep289" \
  --output_path "/data1/large-model-inference/dataset/models_training_dataset/tmp/areal/experiments/checkpoints/root/gsm8k-grpo-stale-3/trial2-attn_backend-triton/default/epoch9epochstep28globalstep289/eval" \
  --max_gen_tokens 32768 \
  --data_names aime24,aime25,amc23,math_500,gpqa_diamond \
  --prompt_type qwen3-think \
  --task math
sleep 60


# CUDA_VISIBLE_DEVICES=4,5,6,7 python eval_and_aggregate.py \
#   --model_path "/home/data/tmp/areal/experiments/checkpoints/root/gsm8k-grpo-stale-2/trial-attn_backend-triton/default/epoch9epochstep28globalstep289" \
#   --output_path "/home/data/tmp/areal/experiments/checkpoints/root/gsm8k-grpo-stale-2/trial-attn_backend-triton/default/epoch9epochstep28globalstep289/eval" \
#   --max_gen_tokens 32768 \
#   --data_names aime24,aime25,amc23,math_500,gpqa_diamond \
#   --prompt_type qwen3-think \
#   --task math
# sleep 60


CUDA_VISIBLE_DEVICES=4,5,6,7 python eval_and_aggregate.py \
  --model_path "/data1/large-model-inference/dataset/models_training_dataset/tmp/areal/experiments/checkpoints/root/gsm8k-grpo-stale-1/trial1-attn_backend-triton/default/epoch9epochstep28globalstep289" \
  --output_path "/data1/large-model-inference/dataset/models_training_dataset/tmp/areal/experiments/checkpoints/root/gsm8k-grpo-stale-1/trial1-attn_backend-triton/default/epoch9epochstep28globalstep289/eval" \
  --max_gen_tokens 32768 \
  --data_names aime24,aime25,amc23,math_500,gpqa_diamond \
  --prompt_type qwen3-think \
  --task math
sleep 60


# Note that the path is a bit different than others for staleness=0.
CUDA_VISIBLE_DEVICES=4,5,6,7 python eval_and_aggregate.py \
  --model_path "/data1/large-model-inference/dataset/models_training_dataset/tmp/areal/experiments/checkpoints/root/gsm8k-grpo-stale-0/trial0-attn_backend-triton/default/epoch9epochstep28globalstep289" \
  --output_path "/data1/large-model-inference/dataset/models_training_dataset/tmp/areal/experiments/checkpoints/root/gsm8k-grpo-stale-0/trial0-attn_backend-triton/default/epoch9epochstep28globalstep289/eval" \
  --max_gen_tokens 32768 \
  --data_names aime24,aime25,amc23,math_500,gpqa_diamond \
  --prompt_type qwen3-think \
  --task math
