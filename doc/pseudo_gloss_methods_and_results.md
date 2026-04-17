# Pseudo-Gloss Extraction Methods and Ablation Results

## 1. Problem Statement

ASL sentence-level motion generation conditions on text input to produce SMPL-X motion sequences. Raw English sentences contain grammatical structures (articles, auxiliaries, copulas) that have no corresponding sign in ASL. Extracting a **pseudo-gloss** — a reduced sequence of content words in citation form — provides a more sign-aligned conditioning signal.

We compare two pseudo-gloss extraction approaches and evaluate their effect on motion generation quality.

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

## 4. Conditioning Modes

Both pseudo-gloss methods feed into the same diffusion model architecture (MotionDiffusionModelV1_CFG) through the `--cond_mode` flag:

| Mode | Conditioning Signal | Encoding |
|------|-------------------|----------|
| `sentence` | Raw English sentence | `condition_proj(CLIP(sentence))` |
| `gloss` | Rule-based pseudo-gloss | `gloss_proj(CLIP(pseudo_gloss))` |
| `sentence_gloss` | Both signals fused | `condition_proj(CLIP(sentence)) + gloss_proj(CLIP(pseudo_gloss))` |

All three modes share the same frozen CLIP text encoder. The `sentence` and `gloss` branches have separate 3-layer MLP projection heads. The `sentence_gloss` mode uses additive fusion of both projected embeddings.

Classifier-Free Guidance (CFG) replaces the fused condition with a learned null embedding with probability `uncond_prob=0.1` during training, and applies `guidance_scale=3.0` at inference.

---

## 5. Experiment Setup

All three experiments use identical hyperparameters:

| Parameter | Value |
|-----------|-------|
| Model | MotionDiffusionModelV1_CFG |
| Prediction type | Epsilon |
| Batch size | 200 |
| Epochs | 150 |
| Learning rate | 1e-4 |
| Rotation repr. | Axis-angle (3D) |
| Body | Upper body only |
| CFG uncond_prob | 0.1 |
| CFG guidance_scale | 3.0 |
| Eval samples | 100 (seed=42) |

---

## 6. Results

### 6.1 Distribution Metrics

| Condition Mode | FID ↓ | Diversity ↑ |
|---------------|-------|-------------|
| `sentence` | **2.2019** | 1.9505 |
| `gloss` | 2.7198 | 2.0524 |
| `sentence_gloss` | 2.5395 | **2.2531** |

### 6.2 Reconstruction Metrics

| Condition Mode | MPJPE_torso ↓ | MPJPE_arms ↓ | MPJPE_lhand ↓ | MPJPE_rhand ↓ | MPJPE_all ↓ | DTW ↓ |
|---------------|-------------|------------|-------------|-------------|-----------|-------|
| `sentence` | **0.1377** | **0.6365** | 0.5360 | 0.5664 | **0.4538** | 2.4239 |
| `gloss` | 0.1736 | 0.6818 | 0.5321 | 0.5672 | 0.4630 | 2.4606 |
| `sentence_gloss` | 0.1595 | 0.6622 | **0.5318** | **0.5645** | 0.4580 | **2.4236** |

### 6.3 Dynamics Metrics

| Condition Mode | Vel. Error ↓ | Accel. Error ↓ | Jerk (Gen) ↓ | Jerk (GT) |
|---------------|------------|--------------|------------|-----------|
| `sentence` | 3.2003 | 5.4086 | 7.7836 | 4.8073 |
| `gloss` | 3.0783 | 5.1968 | 7.2545 | 4.8073 |
| `sentence_gloss` | **2.9531** | **4.9669** | **6.7152** | 4.8073 |

---

## 7. Analysis

### Distribution Quality

The `sentence` mode achieves the best FID (2.20), indicating that the generated motion distribution most closely matches the ground truth when conditioned on raw English. The `gloss`-only mode has the worst FID (2.72), suggesting that the rule-based pseudo-gloss discards some information useful for distribution matching. The `sentence_gloss` fusion partially recovers (2.54) by retaining the full sentence signal alongside the gloss.

The `sentence_gloss` mode produces the highest diversity (2.25), which is desirable — the additive fusion of two complementary signals may encourage the model to explore a broader range of plausible motions.

### Reconstruction Accuracy

For overall joint accuracy (MPJPE_all), `sentence` is best (0.4538), followed closely by `sentence_gloss` (0.4580), with `gloss`-only trailing (0.4630). The gap is small (~2%).

An interesting pattern emerges in the hand joints: `sentence_gloss` achieves the best left-hand (0.5318) and right-hand (0.5645) MPJPE, outperforming even `sentence`-only. This suggests that the pseudo-gloss signal provides complementary information specifically beneficial for hand articulation — which aligns with the linguistic intuition that content words map more directly to hand configurations in ASL.

DTW (temporal alignment) is nearly identical across all three modes (~2.42), indicating that conditioning mode has minimal impact on temporal structure.

### Motion Dynamics

The dynamics metrics reveal the most striking difference: `sentence_gloss` achieves the best velocity error (2.95), acceleration error (4.97), and generated jerk (6.72). The `gloss`-only mode also outperforms `sentence`-only on all dynamics metrics.

This suggests that pseudo-gloss conditioning produces **smoother, more physically plausible motion**. The reduced, sign-aligned conditioning signal may help the model avoid generating spurious high-frequency artifacts that arise from conditioning on grammatically complex English sentences. However, all three modes still generate notably higher jerk than the ground truth (6.7–7.8 vs. 4.8), indicating room for improvement.

### Summary

| Aspect | Best Mode | Interpretation |
|--------|-----------|----------------|
| Distribution matching (FID) | `sentence` | Full English preserves the most information for distribution-level fidelity |
| Diversity | `sentence_gloss` | Dual-signal fusion encourages motion variety |
| Overall joint accuracy | `sentence` | Marginal lead over fusion mode |
| Hand joint accuracy | `sentence_gloss` | Pseudo-gloss helps hand articulation specifically |
| Motion smoothness | `sentence_gloss` | Gloss signal reduces high-frequency artifacts |

The `sentence_gloss` fusion mode offers the best trade-off: it matches `sentence`-only on reconstruction while significantly improving motion dynamics and diversity. The `gloss`-only mode underperforms on distribution and reconstruction metrics, confirming that raw English still carries information (e.g., prosodic cues, discourse context) beyond what the pseudo-gloss captures.

These results motivate the **learning-based voting approach**: rather than relying on fixed rule-based extraction, a trainable gate network can learn which pseudo-gloss words are most informative for motion generation, potentially combining the distribution fidelity of sentence conditioning with the dynamics benefits of gloss conditioning.
