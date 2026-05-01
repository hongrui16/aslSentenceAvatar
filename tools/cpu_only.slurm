#!/bin/bash


#SBATCH --partition=normal                    
#SBATCH --job-name=upd                   
#SBATCH --output=/home/rhong5/research_pro/hand_modeling_pro/aslAvatar/tools/zlog/%x-%N-%j.out  
#SBATCH --error=/home/rhong5/research_pro/hand_modeling_pro/aslAvatar/tools/zlog/%x-%N-%j.err   
#SBATCH --nodes=1
#SBATCH --ntasks=1                 # 需要 4 个独立任务
#SBATCH --mem-per-cpu=30gb          # 每个 CPU 核分配 60GB 内存
#SBATCH --export=ALL 
#SBATCH --time=0-12:00:00            
#SBATCH --cpus-per-task=4            # 每个任务使用 2 个 CPU 核



# to see ID and state of GPUs assigned

module load gnu10                           
module load python

source /home/rhong5/py310Torch/bin/activate

# cd /home/rhong5/research_pro/hand_modeling_pro/aslAvatar/tools


# # python InterHand26M.py --split train  --process_single_hand_img # --use_mini_all_hands_data
# python -u /home/rhong5/research_pro/hand_modeling_pro/aslAvatar/tools/detect_pose.py
# # python InterHand26M.py --split val

cd /scratch/rhong5/dataset/RWTH-PHOENIX-2014
# unzip -q test_poses.zip -d test_poses
# unzip -q val_poses.zip -d val_poses
wget https://www-i6.informatik.rwth-aachen.de/ftp/pub/rwth-phoenix/2016/phoenix-2014-T.v3.tar.gz

tar -xzf phoenix-2014-T.v3.tar.gz

cd /home/rhong5/research_pro/hand_modeling_pro/aslAvatar/tools/
# python InterHand26M.py --split train
# python InterHand26M.py --split val
# python InterHand26M.py --split test

# python -u /home/rhong5/research_pro/hand_modeling_pro/aslAvatar/tools/detect_pose_video.py

sh update_timestamp.sh /scratch/rhong5/dataset/RWTH-PHOENIX-2014

# torchrun --nnodes=${NUM_NODES} --nproc_per_node=${GPUS_PER_NODE} train_VAE.py --debug 

# torchrun --nnodes=1 --nproc_per_node=1 train_VAE.py --batch_size 10

# salloc -p contrib-gpuq -q gpu --nodes=1 --ntasks-per-node=1 --gres=gpu:3g.40gb:1 --mem=40gb -t 0-24:00:00
# salloc -p gpuq -q gpu --nodes=1 --ntasks-per-node=1 --gres=gpu:3g.40gb:1 --mem=40gb -t 0-24:00:00