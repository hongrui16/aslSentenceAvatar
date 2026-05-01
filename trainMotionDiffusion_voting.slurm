#!/bin/bash

#SBATCH --partition=contrib-gpuq
#SBATCH --qos=gpu
#SBATCH --job-name=voting
#SBATCH --output=/home/rhong5/research_pro/hand_modeling_pro/aslSentenceAvatar/zlog/zslurm/%x-%N-%j.out
#SBATCH --error=/home/rhong5/research_pro/hand_modeling_pro/aslSentenceAvatar/zlog/zslurm/%x-%N-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:A100.80gb:1
#SBATCH --mem=40gb
#SBATCH --export=ALL
#SBATCH --time=4-24:00:00
#SBATCH --cpus-per-task=4

module load gnu10
module load python

source /home/rhong5/envs_py310_new/bin/activate
cd /home/rhong5/research_pro/hand_modeling_pro/aslSentenceAvatar

# ---- Step 1: Generate LLM draft glosses (if not cached) ----
echo "==== Generating LLM draft glosses ===="
python tools/generate_llm_draft_gloss.py --modes train test

# ---- Step 2: Train voting diffusion model ----
echo "==== Training voting diffusion model ===="
PORT=$(( (RANDOM << 8 | RANDOM) & 0xFFFF ))
echo "Random port: $PORT"

accelerate launch --num_processes=1 \
    --num_machines=1 \
    --machine_rank=0 \
    --main_process_ip=localhost \
    --main_process_port=$PORT \
    trainMotionDiffusion_voting.py \
        --batch_size 100 --epochs 100 \
        --prediction_type epsilon \
        --uncond_prob 0.1 --guidance_scale 3.0
