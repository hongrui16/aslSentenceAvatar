# Memory Index

- [Project Overview](project_overview.md) — ASL sentence-level avatar generation project using SMPL-X
- [NSA Reproduction](project_nsa_reproduction.md) — `train_NeuralSignActors.py` coexists with `trainMotionDiffusion.py`; dataset stays sparse 53-joint
- [NSA Training Setup](project_training_setup.md) — current run: bs=128, 500 epochs, bf16, A100 80GB
- [Feedback: Scope & Checkpoints](feedback_scope_and_checkpoints.md) — preserve ablation flags; only newest+best ckpts; surgical fixes only
- [Paper Plan](project_paper_plan.md) — Paper 1 (ASL-SignBank-3D attribute completion, AAAI 2027 8/1) + Paper 2 (phonological conditioning for ASL generation)
- [Paper 1 Method Options](project_paper1_method_options.md) — three candidate method angles (A: PhonoFit 3D refinement, B: anomaly detection, C: prototype synthesis); user likes all, not committed
- [CFG + Gloss Training Pipeline](project_cfg_gloss_pipeline.md) — `_cfg` variants add eps prediction, CFG, 3-way cond_mode (sentence/gloss/sentence_gloss); vel_loss off under epsilon
- [Pseudo-Gloss Extraction](project_pseudogloss_extraction.md) — How2SignSMPLXPhonoDataset cache at `./cache/pseudogloss_*.json`; ASL-specific rules (pronouns kept, n't→not, discourse markers dropped). Now the baseline for the paper, not the target method.
- [Pseudo-Gloss Paper Plan](project_pseudogloss_paper_plan.md) — PGG-SLT (NeurIPS 2025) + Sign2GPT (ICLR 2024) prior work; planned LLM-draft + motion-features voting classifier for Paper 2
- [Voting Network Candidates A/B](project_voting_network_design_candidates.md) — **current** text-only voting designs: A (per-token keep/drop gate) or B (ensemble over N LLM samples). Motion does NOT enter the voting network.
- [Voting Network — Prior Design (superseded)](project_voting_network_design.md) — earlier x_t-aware joint voting plan, superseded 2026-04-16
- [Feedback: No API LLMs](feedback_no_api_llm.md) — always use open-source local LLMs, never suggest paid APIs
- [Comparison Experiments](project_comparison_experiments.md) — rule-based (done) + LLM-draft (job 7003805) + voting (job 7002731) ablation results and paths
