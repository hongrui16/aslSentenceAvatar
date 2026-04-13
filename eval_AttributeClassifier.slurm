#!/bin/bash

#SBATCH --partition=contrib-gpuq
#SBATCH --qos=gpu
#SBATCH --job-name=attr-eval
#SBATCH --output=/home/rhong5/research_pro/hand_modeling_pro/aslSentenceAvatar/zlog/zslurm/%x-%N-%j.out
#SBATCH --error=/home/rhong5/research_pro/hand_modeling_pro/aslSentenceAvatar/zlog/zslurm/%x-%N-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:2g.20gb:1
#SBATCH --mem=20gb
#SBATCH --export=ALL
#SBATCH --time=0-01:00:00
#SBATCH --cpus-per-task=4


module load gnu10
module load python

source /home/rhong5/py310Torch/bin/activate
cd /home/rhong5/research_pro/hand_modeling_pro/aslSentenceAvatar

FRAMES_DIR=/scratch/rhong5/dataset/asl_signbank/handcrops_4x224
CKPT_ROOT=/scratch/rhong5/weights/temp_training_weights/aslSentenceAvatar/AttributeClassifier

# ── Eval finger ──
echo "=== Evaluating finger ==="
python train_AttributeClassifier.py \
    --task finger --eval_only \
    --checkpoint $CKPT_ROOT/finger/20260409_152639_job6937347/best_model.pt \
    --frames_dir $FRAMES_DIR \
    --n_frames 4

# ── Eval handshape ──
echo "=== Evaluating handshape ==="
python train_AttributeClassifier.py \
    --task handshape --eval_only \
    --checkpoint $CKPT_ROOT/handshape/20260409_151802_job6937347/best_model.pt \
    --frames_dir $FRAMES_DIR \
    --n_frames 4
