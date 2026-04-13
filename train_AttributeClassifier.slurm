#!/bin/bash

#SBATCH --partition=contrib-gpuq
#SBATCH --qos=gpu
#SBATCH --job-name=attr-cls-v2
#SBATCH --output=/home/rhong5/research_pro/hand_modeling_pro/aslSentenceAvatar/zlog/zslurm/%x-%N-%j.out
#SBATCH --error=/home/rhong5/research_pro/hand_modeling_pro/aslSentenceAvatar/zlog/zslurm/%x-%N-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:3g.40gb:1
#SBATCH --mem=40gb
#SBATCH --export=ALL
#SBATCH --time=1-24:00:00                  # Set to 4 days and 24 hours
#SBATCH --cpus-per-task=8


module load gnu10
module load python

source /home/rhong5/py310Torch/bin/activate
cd /home/rhong5/research_pro/hand_modeling_pro/aslSentenceAvatar

N_FRAMES=8
IMG_SIZE=224
HANDCROPS_DIR=/scratch/rhong5/dataset/asl_signbank/handcrops_${N_FRAMES}x${IMG_SIZE}

# ── Step 1: Extract hand crops (skips already extracted) ──
echo "=== Extracting hand crops ==="
python tools/preextract_signbank_handcrops.py \
    --n_frames $N_FRAMES \
    --img_size $IMG_SIZE \
    --output_dir $HANDCROPS_DIR

# ── Step 2: Train handshape (Flexion) ──
echo "=== Training handshape ==="
python train_AttributeClassifier_v2.py \
    --task handshape \
    --epochs 50 \
    --batch_size 50 \
    --lr 1e-4 \
    --n_frames $N_FRAMES \
    --num_workers 4 \
    --frames_dir $HANDCROPS_DIR

# ── Step 3: Train finger (Selected Fingers) ──
echo "=== Training finger ==="
python train_AttributeClassifier_v2.py \
    --task finger \
    --epochs 50 \
    --batch_size 50 \
    --lr 1e-4 \
    --n_frames $N_FRAMES \
    --num_workers 4 \
    --frames_dir $HANDCROPS_DIR
