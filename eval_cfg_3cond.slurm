#!/bin/bash

#SBATCH --partition=contrib-gpuq
#SBATCH --qos=gpu
#SBATCH --job-name=evalCFG3
#SBATCH --output=/home/rhong5/research_pro/hand_modeling_pro/aslSentenceAvatar/zlog/zslurm/%x-%N-%j.out
#SBATCH --error=/home/rhong5/research_pro/hand_modeling_pro/aslSentenceAvatar/zlog/zslurm/%x-%N-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:2g.20gb:1
#SBATCH --mem=40gb
#SBATCH --export=ALL
#SBATCH --time=0-12:00:00
#SBATCH --cpus-per-task=6

module load gnu10
module load python

source /home/rhong5/py310Torch/bin/activate
cd /home/rhong5/research_pro/hand_modeling_pro/aslSentenceAvatar

CKPT_BASE=/scratch/rhong5/weights/temp_training_weights/aslSentenceAvatar/ASLSenAvatar_v1_cfg/How2SignSMPLX

# ── Experiment 1: sentence ────────────────────────────────────────────────────
echo "========================================"
echo "  Evaluating: cond_mode=sentence"
echo "========================================"
python infer_eval/generate_how2sign_smplx_param_cfg.py \
    --checkpoint ${CKPT_BASE}/20260416_145045_job6997838/best_model.pt \
    --use_upper_body \
    --compute_metrics \
    --num_samples 100 \
    --seed 42

# ── Experiment 2: gloss ──────────────────────────────────────────────────────
echo "========================================"
echo "  Evaluating: cond_mode=gloss"
echo "========================================"
python infer_eval/generate_how2sign_smplx_param_cfg.py \
    --checkpoint ${CKPT_BASE}/20260416_151940_job6997858/best_model.pt \
    --use_upper_body \
    --compute_metrics \
    --num_samples 100 \
    --seed 42

# ── Experiment 3: sentence_gloss ─────────────────────────────────────────────
echo "========================================"
echo "  Evaluating: cond_mode=sentence_gloss"
echo "========================================"
python infer_eval/generate_how2sign_smplx_param_cfg.py \
    --checkpoint ${CKPT_BASE}/20260416_152442_job6997864/best_model.pt \
    --use_upper_body \
    --compute_metrics \
    --num_samples 100 \
    --seed 42

echo "========================================"
echo "  All 3 experiments done!"
echo "========================================"
