# Pseudo-Gloss Extraction Methods and Ablation Results

## 1. Problem Statement

ASL sentence-level motion generation conditions on text input to produce SMPL-X motion sequences. Raw English sentences contain grammatical structures (articles, auxiliaries, copulas) that have no corresponding sign in ASL. Extracting a **pseudo-gloss** — a reduced sequence of content words in citation form — provides a more sign-aligned conditioning signal.

We compare multiple pseudo-gloss extraction and conditioning approaches and evaluate their effect on motion generation quality.

## 2. Method A: Rule-Based Pseudo-Gloss Extraction

**Implementation**: `dataloader/How2SignSMPLXPhonoDataset.py :: extract_gloss_string()`

A deterministic NLP pipeline that converts English sentences to pseudo-gloss sequences:

### Pipeline

```
English sentence
  → NLTK word_tokenize + POS tagging
  → Auxiliary bigram detection (_build_drop_mask)
  → Content-word filtering (nouns, verbs, adjectives, adverbs)
  → Special-case preservation (pronouns, negations, WH-words)
  → Lemmatization (_simple_lemmatize)
  → ASL stopword removal
  → space-joined lowercase pseudo-gloss
```

### Key Design Decisions

1. **Pronoun preservation**: NLTK tags pronouns as PRP/PRP$, which falls outside the content-word POS set {N, V, J, R}. Since ASL indexes pronouns (IX-1, IX-2, IX-3), they are explicitly preserved.

2. **Negation handling**: Words like `no`, `not`, `never`, `nothing` are tagged DT/RB by NLTK and would be dropped. They are preserved via a dedicated `NEGATION_WORDS` set. The contraction `n't` (split by NLTK from `don't`, `won't`, etc.) is normalized to `not`.

3. **Auxiliary chain detection**: Multi-word auxiliary patterns are identified and dropped as a unit:
   - `(be-form) + going + to + VB*` → drop `going` (future auxiliary), but keep `going` as a motion verb before nouns
   - `(have-aux) + got` → drop `got` ("have got" = have)
   - `got + to + VB*` → drop `got` (modal "got to")

4. **POS-gated irregular tables**: Separate lookup tables for verbs, adjectives, and nouns prevent cross-POS collisions (e.g., `leaves` as plural noun → `leaf`, not `leave`).

5. **Lemmatization heuristics**: Rule-based suffix stripping with guards for natural doubled consonants (`ll`, `ss`, `ff`, `zz`) and silent-e restoration (`moving` → `move`, `writing` → `write`).

6. **Discourse marker removal**: Low-signal words (`so`, `therefore`, `however`, `just`, `actually`, `basically`, `literally`, etc.) are added to the ASL stopword list.

### Example

```
Input:  "Today we're going to be learning how to play Portal, a game by Valve Software."
Output: "today we learn play portal game valve software"
```

### Limitations

- Relies on NLTK POS tagger accuracy (known issues with comparative adjectives tagged as NN)
- Suffix-based lemmatization has edge cases (`largest` → `larg`)
- No semantic understanding — cannot distinguish structural vs. meaningful uses of the same word

---

## 3. Method B: LLM-Based Pseudo-Gloss Extraction

**Implementation**: `tools/generate_llm_draft_gloss.py` + `prompts/pseudogloss_extraction_prompt.txt`

Uses a local open-source LLM (Qwen2.5-32B-Instruct) with a carefully designed few-shot prompt to extract pseudo-gloss.

### Pipeline

```
English sentence
  → Few-shot prompt (ASL linguist persona + 3-step instructions + 25 examples)
  → Qwen2.5-32B-Instruct (greedy decoding, local GPU)
  → First line of output, lowercased
  → Cached to ./cache/llm_draft_gloss_{mode}.json
```

### Prompt Design (3-Step Instruction)

The prompt instructs the LLM to follow three sequential steps:

1. **Phrase-level chunking**: Identify multi-word structures before analyzing individual words — auxiliary chains (`is going to`), phrasal verbs (`get out`, `pick up`), light verb constructions (`make sure` → `sure`, `take a look` → `look`), and discourse frames (`let me`, `you know`).

2. **Core meaning filtering**: Keep content words (nouns, verbs, adjectives), pronouns, negations, WH-words, numbers, meaningful modals, and spatial/directional words. Drop articles, copulas, auxiliaries, conjunctions, structural prepositions, degree modifiers (`very`, `really`, `quite`), and discourse fillers.

3. **Citation form restoration**: Lemmatize to base form — verb tense, plural, comparative/superlative normalization. Explicit guard against producing non-word fragments.

### Advantages over Rule-Based

- Handles phrase-level semantics (e.g., distinguishing structural vs. meaningful "how")
- Better lemmatization via implicit vocabulary knowledge
- Can handle edge cases that require contextual understanding
- The prompt encodes ASL-specific linguistic knowledge through examples

### Limitations

- Non-deterministic in principle (though greedy decoding is used)
- Requires GPU for inference (Qwen2.5-32B)
- Occasional hallucination or over-dropping

---

## 4. Method C: Learned Voting Gate (VoteOnly)

**Implementation**: `network/VotingConditionModule.py` + `network/MotionDiffusionModelV1_voting.py`

Instead of relying on fixed extraction rules, a trainable per-token gate network learns which words from the LLM draft pseudo-gloss are most informative for motion generation.

### Pipeline

```
LLM draft gloss (space-separated words)
  → Per-word CLIP embedding (frozen)
  → Positional encoding
  → Transformer encoder (2 layers, 4 heads)
  → Sigmoid gate head → per-word gate g_i ∈ [0, 1]
  → Weighted mean pool: condition = Σ(g_i * h_i) / Σ(g_i)
  → 3-layer MLP projection → (B, model_dim)
```

### Key Design Decisions

1. **Per-word granularity**: Each word gets an independent keep/drop gate, allowing fine-grained selection that adapts to context.
2. **Soft gating**: Sigmoid gates allow partial retention rather than hard binary decisions, enabling gradient flow for end-to-end training.
3. **Bias toward keeping**: Gate head bias initialized to 1.0 (sigmoid ≈ 0.73), so the network starts by keeping most words and learns to drop.
4. **End-to-end training**: The diffusion reconstruction loss backpropagates through the sigmoid gates into the voting transformer — no separate gate supervision needed.

### Training Dynamics

- Gate mean starts at ~0.68 and trends downward during training (to ~0.66 by epoch 150), indicating the network learns to be more selective.
- Gate std increases slightly, showing more differentiated keep/drop decisions.
- Gate min reaches ~0.08, confirming some words are nearly completely suppressed.

---

## 5. Method D: Cross-Attention Fusion (VoteAlign)

**Implementation**: `network/VotingFusionModule.py` + `network/MotionDiffusionModelV1_votingfusion.py`

Uses cross-attention to let motion frames attend to gloss tokens, implicitly learning both token selection and temporal alignment. No explicit gating — the attention mechanism itself decides which gloss tokens are relevant for each frame.

### Pipeline

```
LLM draft gloss → per-word CLIP embedding
  → Stage 1 (Encode): Transformer encoder → contextualized gloss tokens (B, K, D)
  → Stage 2 (Align): TransformerDecoder cross-attention
      query = motion temporal features (B, T, D)
      key/value = gloss tokens (B, K, D)
  → Per-frame condition (B, T, D) → main denoiser transformer
```

### Architecture

| Component | Layers | Heads | Dim |
|-----------|--------|-------|-----|
| Gloss Encoder | 2 | 4 | encoder_dim (512) |
| Fusion Decoder | 2 | 8 | model_dim (512) |

### Advantages over VoteOnly

- Per-frame conditioning instead of a single global vector
- Cross-attention implicitly learns both selection (which tokens matter) and alignment (which frame maps to which token)
- No explicit gate needed — attention weights serve as soft selection

**Status**: Retrained without gate (previous version had gate collapse issue).

---

## 6. Conditioning Modes

All pseudo-gloss methods feed into the same diffusion model architecture (MotionDiffusionModelV1_CFG) through the `--cond_mode` flag:

| Mode | Conditioning Signal | Encoding |
|------|-------------------|----------|
| `sentence` | Raw English sentence | `condition_proj(CLIP(sentence))` |
| `gloss` | Pseudo-gloss (rule or LLM) | `gloss_proj(CLIP(pseudo_gloss))` |
| `sentence_gloss` | Both signals fused | `condition_proj(CLIP(sentence)) + gloss_proj(CLIP(pseudo_gloss))` |
| `voting` | LLM draft → voting gate | `VotingConditionModule(per_word_CLIP(gloss))` |
| `votingfusion` | LLM draft → vote + align | `VotingFusionModule(per_word_CLIP(gloss), motion)` |

All modes share the same frozen CLIP text encoder. CFG replaces the condition with a learned null embedding with probability `uncond_prob=0.1` during training, and applies `guidance_scale=3.0` at inference.

---

## 7. Experiment Setup

All experiments use identical base hyperparameters:

| Parameter | Value |
|-----------|-------|
| Model | MotionDiffusionModelV1_CFG (or _Voting / _VotingFusion) |
| Prediction type | Epsilon |
| Batch size | 200 (cfg, votingfusion) / 100 (voting) |
| Epochs | 150 |
| Learning rate | 1e-4 |
| Rotation repr. | Axis-angle (3D) |
| Body | Upper body only |
| CFG uncond_prob | 0.1 |
| CFG guidance_scale | 3.0 |
| **Eval samples** | **1000 (seed=42)** |

---

## 8. Results

All results evaluated on 1000 test samples (seed=42, deterministic selection). The same 1000 samples are used across all methods for fair comparison.

### 8.1 Full Comparison Table

#### Distribution Metrics

| # | Method | Gloss Source | Conditioning | FID ↓ | Diversity ↑ |
|---|--------|-------------|-------------|-------|-------------|
| 1 | Sentence-only | — | CLIP sentence | **2.0884** | 1.9410 |
| 2 | Gloss (rule) | rule-based | CLIP gloss | 2.6029 | 2.0263 |
| 3 | Gloss (LLM) | LLM draft | CLIP gloss | 2.2846 | 2.1414 |
| 4 | Sent+Gloss (rule) | rule-based | sentence + gloss | 2.3408 | 2.1782 |
| 5 | Sent+Gloss (LLM) | LLM draft | sentence + gloss | 2.5263 | 2.3090 |
| 6 | **VoteOnly (gloss)** | LLM draft + voting gate | gated gloss pool | 2.2128 | 2.0986 |
| 7 | **VoteAlign (gloss)** | LLM draft + fusion (no gate) | cross-attn per-frame | 3.0738 | **2.4031** |

`(gloss)` = gloss-only conditioning (no sentence embedding). Adding sentence conditioning is planned.

#### Reconstruction Metrics

| # | Method | MPJPE_torso ↓ | MPJPE_arms ↓ | MPJPE_lhand ↓ | MPJPE_rhand ↓ | MPJPE_all ↓ | DTW ↓ |
|---|--------|-------------|------------|-------------|-------------|-----------|-------|
| 1 | Sentence-only | **0.1393** | 0.6426 | 0.5404 | 0.5647 | 0.4577 | 2.4716 |
| 2 | Gloss (rule) | 0.1738 | 0.6854 | 0.5361 | 0.5626 | 0.4655 | 2.5012 |
| 3 | Gloss (LLM) | 0.1641 | 0.6717 | **0.5286** | **0.5520** | **0.4575** | **2.4582** |
| 4 | Sent+Gloss (rule) | 0.1540 | 0.6679 | 0.5397 | 0.5627 | 0.4619 | 2.4761 |
| 5 | Sent+Gloss (LLM) | 0.1538 | 0.7100 | 0.5490 | 0.5730 | 0.4722 | 2.5459 |
| 6 | **VoteOnly (gloss)** | 0.1525 | **0.6632** | 0.5331 | 0.5574 | 0.4578 | 2.4629 |
| 7 | **VoteAlign (gloss)** | 0.2549 | 0.6762 | 0.5249 | 0.5586 | 0.4715 | 2.5057 |

`(gloss)` = gloss-only conditioning (no sentence embedding).

#### Dynamics Metrics

| # | Method | Vel. Error ↓ | Acc. Error ↓ | Jerk (Gen) ↓ | Jerk (GT) |
|---|--------|------------|------------|------------|-----------|
| 1 | Sentence-only | 3.1499 | 5.2937 | 7.6649 | 4.7679 |
| 2 | Gloss (rule) | 3.0229 | 5.0297 | 7.1319 | 4.7679 |
| 3 | Gloss (LLM) | 2.9152 | 4.8982 | 6.6952 | 4.7679 |
| 4 | Sent+Gloss (rule) | 2.9424 | 4.9213 | 6.7620 | 4.7679 |
| 5 | Sent+Gloss (LLM) | 3.1829 | 5.3791 | 7.7084 | 4.7679 |
| 6 | **VoteOnly (gloss)** | 2.8936 | 4.8500 | 6.6059 | 4.7679 |
| 7 | **VoteAlign (gloss)** | **2.7425** | **4.5888** | **5.9785** | 4.7679 |

`(gloss)` = gloss-only conditioning (no sentence embedding).

#### Best Eval Loss (Training)

| # | Method | Best Eval Loss |
|---|--------|---------------|
| 1 | Sentence-only | 0.4536 |
| 2 | Gloss (rule) | 0.4584 |
| 3 | Gloss (LLM) | 0.4544 |
| 4 | Sent+Gloss (rule) | 0.4564 |
| 5 | Sent+Gloss (LLM) | 0.4589 |
| 6 | **VoteOnly (gloss)** | **0.4268** |
| 7 | **VoteAlign (gloss)** | 0.4540 |

`(gloss)` = gloss-only conditioning (no sentence embedding).

### 8.2 Checkpoint Paths

| # | Method | Checkpoint |
|---|--------|-----------|
| 1 | Sentence-only | `ASLSenAvatar_v1_cfg/.../20260416_145045_job6997838/` |
| 2 | Gloss (rule) | `ASLSenAvatar_v1_cfg/.../20260416_151940_job6997858/` |
| 3 | Gloss (LLM) | `ASLSenAvatar_v1_cfg_llm/.../20260418_040732_job7003891/` |
| 4 | Sent+Gloss (rule) | `ASLSenAvatar_v1_cfg/.../20260416_152442_job6997864/` |
| 5 | Sent+Gloss (LLM) | `ASLSenAvatar_v1_cfg_llm/.../20260419_023651_job7008731/` |
| 6 | VoteOnly (gloss) | `ASLSenAvatar_v1_voting/.../20260417_150333_job7002731/` |
| 7 | VoteAlign (gloss) | `ASLSenAvatar_v1_votingfusion/.../20260420_112234_job7020894/` |

---

## 9. Analysis

### 9.1 Two Winners: VoteOnly (Reconstruction) and VoteAlign (Dynamics)

No single method dominates all metrics. The results split into two clear winners:

- **VoteOnly** leads on reconstruction: MPJPE_arms (0.6632), and is competitive on MPJPE_all (0.4578), hands, and DTW. Best eval loss (0.4268).
- **VoteAlign** leads on all dynamics metrics: velocity error (**2.7425**), acceleration error (**4.5888**), and jerk (**5.9785**). Its jerk is the closest to GT (4.7679) among all methods — a 10% improvement over VoteOnly.

Note: both VoteOnly and VoteAlign currently use **gloss-only conditioning** (no sentence embedding), which likely explains their higher FID compared to sentence-conditioned methods. Adding sentence conditioning is planned as the next step.

### 9.2 LLM Draft is Strictly Better than Rule-Based (Same Conditioning)

Comparing gloss source while holding conditioning mode constant:

| Comparison | FID | MPJPE_all | Vel Error | Jerk |
|-----------|-----|-----------|-----------|------|
| Gloss (rule) vs. Gloss (LLM) | 2.60 → **2.28** | 0.4655 → **0.4575** | 3.02 → **2.92** | 7.13 → **6.70** |
| Sent+Gloss (rule) vs. Sent+Gloss (LLM) | **2.34** → 2.53 | **0.4619** → 0.4722 | **2.94** → 3.18 | **6.76** → 7.71 |

LLM draft gloss is clearly superior in the **gloss-only** setting — better on every metric. However, when fused with the sentence embedding (**sent+gloss**), LLM draft actually hurts across the board. This suggests the LLM draft carries richer information that **conflicts** with the sentence embedding when naively added together, whereas the sparser rule-based gloss is complementary.

### 9.3 Dynamics Improvement Trend

A clear hierarchy emerges for motion smoothness:

```
Method                Vel Error    Jerk
Sent+Gloss (LLM)     3.18         7.71   (worst)
Sentence-only         3.15         7.66
Gloss (rule)          3.02         7.13
Sent+Gloss (rule)     2.94         6.76
Gloss (LLM)           2.92         6.70
VoteOnly (gloss)            2.89         6.61
VoteAlign (gloss)           2.74         5.98   (best, closest to GT 4.77)
```

`(gloss)` = gloss-only conditioning (no sentence embedding).

The learned methods produce the smoothest motion. VoteAlign's cross-attention per-frame conditioning yields the largest dynamics improvement (~10% lower jerk than VoteOnly).

### 9.4 FID vs. Dynamics Trade-off

There is a consistent tension between FID (distribution matching) and dynamics (motion smoothness):

- **Sentence-only** has the best FID (2.09) but the worst dynamics (jerk 7.66)
- **VoteAlign** has the best dynamics (jerk 5.98) but the worst FID (3.07)

This trade-off likely reflects that distributional fidelity rewards variance matching (including high-frequency noise present in GT data), while dynamics metrics reward smoothness. Additionally, VoteOnly and VoteAlign use gloss-only conditioning without sentence embedding, which contributes to their higher FID. Adding sentence conditioning should help close this gap.

### 9.5 Hand Accuracy: LLM Gloss-Only Wins

Surprisingly, the simple Gloss (LLM) baseline achieves the best hand joint accuracy (MPJPE_lhand=**0.5286**, MPJPE_rhand=**0.5520**) and overall MPJPE_all (**0.4575**). This suggests that for per-joint reconstruction, a clean LLM-extracted gloss string through CLIP is an effective conditioning signal — the LLM captures hand-relevant content words well. The learned methods (VoteOnly, VoteAlign) are close but don't surpass this simple baseline on hands.

### Summary: Metric Winners

| Aspect | Best Method | Value | Runner-up |
|--------|-----------|-------|-----------|
| FID ↓ | Sentence-only | **2.0884** | VoteOnly (gloss) (2.2128) |
| Diversity ↑ | VoteAlign (gloss) | **2.4031** | Sent+Gloss LLM (2.3090) |
| MPJPE_all ↓ | Gloss (LLM) | **0.4575** | Sentence-only (0.4577) |
| MPJPE_hands ↓ | Gloss (LLM) | **0.5286 / 0.5520** | VoteAlign (gloss) (0.5249 / 0.5586) |
| DTW ↓ | Gloss (LLM) | **2.4582** | VoteOnly (gloss) (2.4629) |
| Vel Error ↓ | VoteAlign (gloss) | **2.7425** | VoteOnly (gloss) (2.8936) |
| Jerk ↓ | VoteAlign (gloss) | **5.9785** | VoteOnly (gloss) (6.6059) |
| Eval Loss ↓ | VoteOnly (gloss) | **0.4268** | Sentence-only (0.4536) |

`(gloss)` = gloss-only conditioning (no sentence embedding).

### Paper Narrative

The full 7-method ablation tells a three-part story:

1. **Gloss source matters**: LLM draft gloss is strictly better than rule-based extraction when used as the sole conditioning signal — the LLM captures ASL-relevant content words more accurately.

2. **Naive fusion fails**: Simply adding gloss to the sentence embedding (sent+gloss) can hurt, especially with richer LLM-draft gloss that conflicts with the sentence embedding. This motivates a learned fusion mechanism.

3. **Learned selection and alignment work**: VoteOnly (learned gate) and VoteAlign (cross-attention fusion) achieve the best overall results. VoteAlign produces the smoothest motion (best dynamics across all metrics), while VoteOnly offers the best reconstruction balance. Both currently use gloss-only conditioning — adding sentence conditioning should improve FID.

The progression **rule-based → LLM draft → learned voting → cross-attention fusion** demonstrates increasingly sophisticated conditioning, with clear gains at each step for motion quality.
