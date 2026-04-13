#!/bin/bash

#SBATCH --partition=contrib-gpuq                    # need to set 'gpuq' or 'contrib-gpuq' partition
#SBATCH --qos=gpu                           # need to select 'gpu' QOS or other relevant QOS
#SBATCH --job-name=trans                  # Job name
#SBATCH --output=/home/rhong5/research_pro/hand_modeling_pro/aslSentenceAvatar/zlog/zslurm/%x-%N-%j.out   # Output file
#SBATCH --error=/home/rhong5/research_pro/hand_modeling_pro/aslSentenceAvatar/zlog/zslurm/%x-%N-%j.err    # Error file
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1                # One task per GPU
#SBATCH --gres=gpu:A100.80gb:1             # Request 3 A100 GPUs
#SBATCH --mem=40gb                         # Memory per node
#SBATCH --export=ALL
#SBATCH --time=4-24:00:00                  # Set to 4 days and 24 hours
#SBATCH --cpus-per-task=6                  # Number of CPU cores per task




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
    train_backtrans.py  --use_upper_body --epochs 60 \
        --freeze_epochs 25 \
        --encoder_lr 1e-4 \
        --decoder_lr 1e-5 \
        --token_drop 0.3 \
        --n_pool 24






# ASLLVD_Skeleton3D SignBank_SMPLX WLASL_SMPLX


# salloc -p contrib-gpuq -q gpu --nodes=1 --ntasks-per-node=1 --gres=gpu:3g.40gb:1 --mem=40gb -t 0-24:00:00
# salloc -p gpuq -q gpu --nodes=1 --ntasks-per-node=1 --gres=gpu:A100.40gb:1 --mem=40gb -t 0-24:00:00
# salloc -p gpuq -q gpu --nodes=1 --ntasks-per-node=1 --gres=gpu:3g.40gb:1 --mem=40gb -t 0-24:00:00
# salloc -p gpuq -q gpu --nodes=1 --ntasks-per-node=1 --gres=gpu:2g.20gb:1 --mem=40gb -t 0-24:00:00
# salloc -p contrib-gpuq -q gpu --nodes=1 --ntasks-per-node=1 --gres=gpu:2g.20gb:1 --mem=40gb -t 0-24:00:00