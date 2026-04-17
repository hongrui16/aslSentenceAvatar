#!/bin/bash

#SBATCH --partition=contrib-gpuq
#SBATCH --qos=gpu
#SBATCH --job-name=nsa2
#SBATCH --output=/home/rhong5/research_pro/hand_modeling_pro/aslSentenceAvatar/zlog/zslurm/%x-%N-%j.out
#SBATCH --error=/home/rhong5/research_pro/hand_modeling_pro/aslSentenceAvatar/zlog/zslurm/%x-%N-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:A100.80gb:1
#SBATCH --mem=60gb
#SBATCH --export=ALL
#SBATCH --time=4-24:00:00
#SBATCH --cpus-per-task=6




module load gnu10
module load python



source /home/rhong5/py310Torch/bin/activate
cd /home/rhong5/research_pro/hand_modeling_pro/aslSentenceAvatar

PORT=$(( (RANDOM << 8 | RANDOM) & 0xFFFF ))
echo "Random port: $PORT"




# accelerate launch --num_processes=1 \
#     --num_machines=1 \
#     --machine_rank=0 \
#     --main_process_ip=localhost \
#     --main_process_port=$PORT \
#     train_NeuralSignActors.py --batch_size 128 --epochs 500

accelerate launch --num_processes=1 \
    --num_machines=1 \
    --machine_rank=0 \
    --main_process_ip=localhost \
    --main_process_port=$PORT \
    train_NeuralSignActors.py --batch_size 120 --epochs 500 --use_gloss_condition                   


# Common variants:
#   严格复刻论文(axis-angle, 全身,无表情):
#     train_NeuralSignActors.py --batch_size 96 --epochs 2000
#
#   论文 + 表情分支:
#     train_NeuralSignActors.py --batch_size 96 --epochs 2000 --use_expression
#
#   对比实验(6D + 上半身):
#     train_NeuralSignActors.py --batch_size 96 --epochs 2000 --use_rot6d --use_upper_body
#
#   快速调试:
#     train_NeuralSignActors.py --batch_size 16 --debug
#
#   恢复训练:
#     train_NeuralSignActors.py --batch_size 96 --resume /path/to/best_model.pt


# Interactive allocations:
# salloc -p contrib-gpuq -q gpu --nodes=1 --ntasks-per-node=1 --gres=gpu:A100.80gb:1 --mem=60gb -t 0-24:00:00
# salloc -p gpuq         -q gpu --nodes=1 --ntasks-per-node=1 --gres=gpu:A100.80gb:1 --mem=60gb -t 0-24:00:00
