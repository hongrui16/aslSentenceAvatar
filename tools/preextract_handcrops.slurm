#!/bin/bash

#SBATCH --partition=normal
#SBATCH --job-name=handcrop
#SBATCH --output=/home/rhong5/research_pro/hand_modeling_pro/aslSentenceAvatar/zlog/zslurm/%x-%N-%j.out
#SBATCH --error=/home/rhong5/research_pro/hand_modeling_pro/aslSentenceAvatar/zlog/zslurm/%x-%N-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=30gb
#SBATCH --export=ALL
#SBATCH --time=1-00:00:00
#SBATCH --cpus-per-task=4

module load gnu10
module load python

source /home/rhong5/py310Torch/bin/activate

cd /home/rhong5/research_pro/hand_modeling_pro/aslSentenceAvatar

python -u tools/preextract_signbank_handcrops.py --n_frames 8 --img_size 224
