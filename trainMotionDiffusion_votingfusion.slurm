#!/bin/bash

#SBATCH --partition=contrib-gpuq
#SBATCH --qos=gpu
#SBATCH --job-name=votfus
#SBATCH --output=/home/rhong5/research_pro/hand_modeling_pro/aslSentenceAvatar/zlog/zslurm/%x-%N-%j.out
#SBATCH --error=/home/rhong5/research_pro/hand_modeling_pro/aslSentenceAvatar/zlog/zslurm/%x-%N-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:2g.20gb:1
#SBATCH --mem=40gb
#SBATCH --export=ALL
#SBATCH --time=4-24:00:00
#SBATCH --cpus-per-task=4

module load gnu10
module load python

source /home/rhong5/py310Torch/bin/activate
cd /home/rhong5/research_pro/hand_modeling_pro/aslSentenceAvatar

PORT=$(( (RANDOM << 8 | RANDOM) & 0xFFFF ))
echo "Random port: $PORT"

accelerate launch --num_processes=1 \
    --num_machines=1 \
    --machine_rank=0 \
    --main_process_ip=localhost \
    --main_process_port=$PORT \
    trainMotionDiffusion_votingfusion.py \
        --batch_size 200 \
        --epochs 100 \
        --voting_n_layers 2 \
        --voting_n_heads 4 \
        --fusion_n_layers 2 \
        --fusion_n_heads 8
