#!/bin/bash
#SBATCH --job-name=knowledge_graph_training
#SBATCH --output=output.out
#SBATCH --mem=128G
#SBATCH --gres=gpu:1

source ~/.bashrc
conda activate ecause

cd /home/hgao62/Position-Bias-Mitigation-in-Emotion-Cause-Analysis

# GPU_ID="0"
# MODELS="Qwen/Qwen2.5-7B,meta-llama/Llama-3.1-8B,deepseek-ai/DeepSeek-R1-Distill-Qwen-7B,deepseek-ai/DeepSeek-R1-Distill-Llama-8B"

# MODELS="Qwen/Qwen2.5-7B,meta-llama/Llama-3.1-8B"

# PROMPT_FILE="prompts.txt"
# MAX_NEW_TOKENS=1024

python ad_paedgl.py
    # --gpu_id $GPU_ID \
    # --models $MODELS \
    # --prompt_file $PROMPT_FILE \
    # --max_new_tokens $MAX_NEW_TOKENS