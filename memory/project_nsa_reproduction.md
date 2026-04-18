---
name: Neural Sign Actors reproduction
description: NSA paper (arXiv 2312.02702) strict reproduction baseline in this repo
type: project
originSessionId: 2be47fb7-b67d-4054-9d23-8841007fd70f
---
`train_NeuralSignActors.py` + `network/NeuralSignActorsModel.py` are a faithful reproduction of Neural Sign Actors (Baltatzis et al., arXiv 2312.02702). `trainMotionDiffusion.py` + `MotionDiffusionModelV1/V2` is the user's own prior method — both must coexist and use the same `How2SignSMPLXDataset`.

**Why:** User is building comparison baselines — NSA as faithful reference, own method for ablation/improvement.

**How to apply:**
- Any change to `dataloader/How2SignSMPLXDataset.py` must keep both training paths working. Dataset outputs sparse 53-joint layout (`joint_indices = ALL_53_JOINTS` always); models handle `USE_UPPER_BODY` themselves via `tosave_slices` (V1/V2) or bypass logic (NSA). Do not switch dataset to compact layout.
- Expression (10 blendshape coeffs) is appended at tail when `--use_expression`. NSA slices it at `expr_start = 53 * n_feats`; V1/V2 don't use it.
- NSA flags `--use_rot6d --use_upper_body --use_expression --root_normalize --use_phono_attribute` are **all** kept as ablation dimensions — do not strip them even when "strictly reproducing the paper". Paper defaults correspond to none of these being set.
- Jaw belongs to `REMOVE_INDICES` in `utils/rotation_conversion.py` (removed from `UPPER_BODY_INDICES`); NSA/V1/V2 models do NOT bypass jaw (only bypass LOWER_BODY on `USE_UPPER_BODY`). This intentional asymmetry is fine given sparse dataset layout.
