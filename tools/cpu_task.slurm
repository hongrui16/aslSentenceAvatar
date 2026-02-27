#!/bin/bash
#SBATCH --partition=contrib-gpuq                    # need to set 'gpuq' or 'contrib-gpuq'  partition
#SBATCH --qos=gpu                           # need to select 'gpu' QOS or other relevant QOS
#SBATCH --job-name=upd                   # vqgan, vqvae, vqdiffusion, gaussiandiffusion
#SBATCH --output=./zlog/%x-%N-%j.out   # Output file
#SBATCH --error=./zlog/%x-%N-%j.err    # Error file
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1                # number of cores needed
#SBATCH --mem=20gb                # 100gb for InterHand26M, memory per CORE; total memory is 1 TB (1,000,000 MB), 
#SBATCH --gres=gpu:1g.10gb:1             # Request 3 A100 GPUs
#SBATCH --export=ALL 
#SBATCH --time=2-24:00:00                   # set to 24hr; please choose carefully
#SBATCH --cpus-per-task=4  # Number of CPU cores per task

set echo
umask 0027

# to see ID and state of GPUs assigned
# nvidia-smi

# module load gnu10                           
# module load python

source /home/rhong5/py310Torch/bin/activate
# # source /home/rhong5/env_mmcv/bin/activate
# # cd /home/rhong5/research_pro/hand_modeling_pro/HandPoseSD
# # python train_handposeRegressor.py

cd /home/rhong5/research_pro/hand_modeling_pro/aslAvatar/tools/
# python InterHand26M.py --split train
# python InterHand26M.py --split val
# python InterHand26M.py --split test

# python /home/rhong5/research_pro/hand_modeling_pro/aslAvatar/tools/detect_pose.py

sh update_timestamp.sh /scratch/rhong5/dataset/InterHand

# torchrun --nnodes=${NUM_NODES} --nproc_per_node=${GPUS_PER_NODE} train_VAE.py --debug 

# torchrun --nnodes=1 --nproc_per_node=1 train_VAE.py --batch_size 10

# salloc -p contrib-gpuq -q gpu --nodes=1 --ntasks-per-node=1 --gres=gpu:3g.40gb:1 --mem=40gb -t 0-24:00:00
# salloc -p gpuq -q gpu --nodes=1 --ntasks-per-node=1 --gres=gpu:3g.40gb:1 --mem=40gb -t 0-24:00:00