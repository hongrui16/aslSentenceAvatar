# File Guide

## Training Scripts

| File | Description |
|------|-------------|
| `trainMotionDiffusion.py` | Original MDM training (x_0 prediction). Condition: sentence embedding (single vector). Uses `MotionDiffusionModelV1/V2`. |
| `trainMotionDiffusion_cfg.py` | MDM + epsilon prediction + CFG. Supports 3 condition modes via `--cond_mode`: `sentence`, `gloss` (rule-based pseudo-gloss), `sentence_gloss` (additive fusion). Also supports `--gloss_source {rule, llm_draft}` to switch between rule-based and LLM-drafted glosses. |
| `trainMotionDiffusion_voting.py` | MDM + voting gate (Design A). LLM draft gloss → per-word CLIP embedding → voting transformer → sigmoid gate (keep/drop per word) → weighted mean pool → single condition vector. Gate trained end-to-end via diffusion loss. |
| `trainMotionDiffusion_votingfusion.py` | MDM + voting gate + cross-attention fusion. LLM draft gloss → voting gate → gated gloss tokens → cross-attention with motion temporal features → per-frame condition. Implicitly learns temporal alignment (gloss reordering). |
| `train_NeuralSignActors.py` | Neural Sign Actors (Baltatzis et al.) strict reproduction baseline. |
| `train_backtrans.py` | Back-translation model (motion → text) for quality assessment. |
| `train_AttributeClassifier_v1.py` | Handshape attribute classifier V1 (Paper 1). |
| `train_AttributeClassifier_v2.py` | Handshape attribute classifier V2, improved architecture (Paper 1). |

## Network Modules

| File | Description |
|------|-------------|
| `network/MotionDiffusionModelV1.py` | MDM V1: transformer encoder denoiser, cosine noise schedule (1000 steps), DDIM sampling (50 steps), x_0 prediction. Text condition via CLIP/T5 → single vector prepended as token. |
| `network/MotionDiffusionModelV2.py` | MDM V2: V1 + kinematic GNN encoder that processes joints along the SMPL-X skeleton tree before the transformer. |
| `network/MotionDiffusionModelV1_cfg.py` | MDM V1 + epsilon prediction + classifier-free guidance. Adds `gloss_proj` (separate 3-layer MLP) for gloss conditioning. `sentence_gloss` mode fuses via element-wise addition. Parent class for voting/votingfusion variants. |
| `network/MotionDiffusionModelV2_cfg.py` | MDM V2 + epsilon prediction + CFG (same extensions as V1_cfg). |
| `network/MotionDiffusionModelV1_voting.py` | MDM V1 + voting gate. Overrides `get_condition()` to use `VotingConditionModule`. Per-word CLIP embedding → voting transformer → gate → weighted mean pool → single condition vector (B, D). |
| `network/MotionDiffusionModelV1_votingfusion.py` | MDM V1 + voting gate + cross-attention fusion. Overrides `denoise()` to use `VotingFusionModule`. Gated gloss tokens serve as memory for TransformerDecoder cross-attention with motion queries → per-frame condition (B, T, D). |
| `network/VotingConditionModule.py` | Per-gloss voting gate. Input: word embeddings (B, K, D) → transformer encoder → sigmoid gate per word → weighted mean pool → single condition vector. Used by V1_voting. |
| `network/VotingFusionModule.py` | Voting gate + cross-attention fusion. Stage 1 (vote): same gate as VotingConditionModule but outputs gated token sequence. Stage 2 (fuse): TransformerDecoder where motion tokens cross-attend to gated gloss tokens → per-frame condition. Used by V1_votingfusion. |
| `network/NeuralSignActorsModel.py` | NSA paper reproduction model. |
| `network/PhonoSignActorsModel.py` | NSA + phonological attribute conditioning (Paper 2 variant). |
| `network/BackTranslationModel.py` | Transformer encoder + Perceiver temporal pooling + T5 decoder for motion → text. |
| `network/HandshapeClassifier.py` | Handshape attribute classifier V1 (Paper 1). |
| `network/HandshapeClassifierV2.py` | Handshape attribute classifier V2 (Paper 1). |

## Dataloaders

| File | Description |
|------|-------------|
| `dataloader/How2SignSMPLXDataset.py` | Base dataset. Loads per-frame SMPL-X pkl (53 joints × 3 axis-angle = 159-D). Returns `(motion, sentence, sentence, length)`. Supports rot6d (318-D) and upper-body-only modes. Sequences padded/sampled to 200 frames. |
| `dataloader/How2SignSMPLXPhonoDataset.py` | Extends base dataset. Adds rule-based pseudo-gloss extraction with disk cache (`cache/pseudogloss_*.json`). Returns `(motion, sentence, gloss_string, length)`. |
| `dataloader/How2SignSMPLXVotingDataset.py` | Extends base dataset. Loads LLM-drafted pseudo-gloss from pre-computed cache (`cache/llm_draft_gloss_*.json`). Returns `(motion, sentence, llm_draft_gloss, length)`. Same output format as PhonoDataset. |
| `dataloader/SignBankHandshapeDataset.py` | ASL-SignBank dataset for handshape attribute classification (Paper 1). |

## Config

| File | Description |
|------|-------------|
| `config.py` | `BaseConfig` and dataset-specific configs (`How2Sign_SMPLX_Config`, etc.). Key flags: `--use_rot6d`, `--use_upper_body`, `--model_version`, `--text_encoder_type`. |

## Utils

| File | Description |
|------|-------------|
| `utils/rotation_conversion.py` | Axis-angle ↔ 6D rotation conversions, joint group definitions (`UPPER_BODY_INDICES`, `REMOVE_INDICES`), `get_joint_slices()` for per-group loss weighting. |
| `utils/utils.py` | `plot_training_curves()`, `backup_code()`, `collate_fn()`, `create_padding_mask()`. |
| `utils/renders.py` | SMPL-X mesh rendering utilities (pyrender). |
| `utils/model_free_metrics.py` | FID, MPJPE, DTW, diversity, velocity/acceleration/jerk metrics for evaluation. |

## Tools

| File | Description |
|------|-------------|
| `tools/generate_llm_draft_gloss.py` | Generates LLM draft pseudo-gloss for all How2Sign sentences using a local open-source LLM. Outputs `cache/llm_draft_gloss_{train,val,test}.json`. Must run before voting/votingfusion training. |
| `tools/analyze_gloss_distribution.py` | Analyzes pseudo-gloss extraction quality — word frequencies, coverage, POS distributions. |
| `tools/analyze_seq_lengths.py` | Analyzes motion sequence length distributions. |
| `tools/calculate_dist.py` | Calculates distance metrics between generated and GT motions. |
| `tools/concat_keyframes.py` | Concatenates keyframe visualizations. |
| `tools/count_how2sign_word.py` | Word frequency analysis on How2Sign captions. |
| `tools/detect_pose.py` | Pose detection pipeline (HRNet/ViTPose). |
| `tools/detect_pose_recheck.py` | Re-checks failed pose detections. |
| `tools/detect_pose_video.py` | Video-level pose detection. |
| `tools/diagnostic_smplx_axes.py` | Diagnostic visualization for SMPL-X axis conventions. |
| `tools/filter_files.py` | Filters/cleans dataset files. |
| `tools/generate_synthetic_smplx.py` | Generates synthetic SMPL-X data for testing. |
| `tools/gen_missing.py` | Identifies missing generated samples. |
| `tools/gen_retry.py` | Retries failed generation runs. |
| `tools/preextract_signbank_frames.py` | Pre-extracts video frames from SignBank videos. |
| `tools/preextract_signbank_handcrops.py` | Pre-extracts hand crop images from SignBank. |
| `tools/render_mesh_signbank.py` | Renders SMPL-X meshes for SignBank samples. |
| `tools/render_mesh_wlasl.py` | Renders SMPL-X meshes for WLASL samples. |
| `tools/render_synthetic_smplx.py` | Renders synthetic SMPL-X data. |
| `tools/verify_fitting_param.py` | Verifies SMPL-X fitting parameter quality. |
| `tools/visualize_comparison.py` | Side-by-side comparison visualizations (GT vs generated). |
| `tools/vis_wlasl_examples.py` | Visualizes WLASL example videos. |
| `tools/wlasl_diag.py` | WLASL dataset diagnostics. |
| `tools/wlasl_split.py` | WLASL train/val/test split generation. |

## SLURM Scripts

| File | Description |
|------|-------------|
| `trainMotionDiffusion.slurm` | Runs `trainMotionDiffusion_cfg.py` (used for 3-way cond_mode comparison with rule-based gloss). |
| `trainMotionDiffusion_llm3.slurm` | Runs `trainMotionDiffusion_cfg.py` with `--gloss_source llm_draft` for 3 cond_modes sequentially. |
| `trainMotionDiffusion_voting.slurm` | Runs `trainMotionDiffusion_voting.py` (voting gate only). |
| `trainMotionDiffusion_votingfusion.slurm` | Runs `trainMotionDiffusion_votingfusion.py` (voting + cross-attention fusion). |
| `trainMotionDiffusion2.slurm` | Alternative MDM training config. |
| `trainMotionDiffusion3.slurm` | Alternative MDM training config. |
| `train_NeuralSignActors.slurm` | NSA reproduction training. |
| `train_NeuralSignActors_2.slurm` | NSA alternative config. |
| `train_backtrans.slurm` | Back-translation model training. |
| `train_AttributeClassifier.slurm` | Handshape classifier training. |
| `eval_cfg_3cond.slurm` | Evaluation for the 3-way cfg comparison experiments. |

## Model Evolution (Paper 2)

```
V1_cfg (sentence)          ← baseline: sentence-only conditioning
V1_cfg (gloss)             ← baseline: rule-based gloss conditioning
V1_cfg (sentence_gloss)    ← baseline: additive fusion
V1_voting                  ← proposed: LLM draft + learnable per-token gate
V1_votingfusion            ← proposed: voting + cross-attention temporal fusion
```
