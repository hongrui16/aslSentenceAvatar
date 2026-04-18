---
name: Next step — add phonology attributes from ASL-SignBank
description: After current ablation runs finish, add SignBank phonology attributes (handshape/location/movement) as additional conditioning to the votingfusion pipeline.
type: project
---

User stated (2026-04-18): after all current experiments complete (rule-based cfg, LLM-draft cfg, voting, votingfusion), the next step is to incorporate ASL-SignBank phonology attributes into the pipeline.

## What this means

ASL-SignBank provides per-gloss phonological annotations: handshape, location, selected fingers, flexion, movement type, etc. These can serve as structured conditioning beyond the raw gloss text.

## Open design questions (not yet discussed)

1. How to map How2Sign sentence glosses to SignBank entries (gloss string matching? fuzzy matching?)
2. Which attributes to use (handshape alone? full phonology vector?)
3. Where to inject: as extra features concatenated to gloss embeddings before voting? As a separate conditioning branch? As part of cross-attention keys?
4. How to handle glosses not found in SignBank (fallback to text-only?)
5. Relationship to Paper 1 (ASL-SignBank-3D attribute classifiers) — can Paper 1's trained classifiers provide phonology embeddings?

## Connection to existing work

- `--use_phono_attribute` flag already exists in `train_NeuralSignActors.py` and `config.py` (wired but not yet used in the voting/votingfusion pipeline)
- Paper 1 trains attribute classifiers on SignBank data → could produce phonology embeddings for Paper 2
- This would be the phonological conditioning angle mentioned in [Paper Plan](project_paper_plan.md)

**Why:** User's goal is a top-venue paper. Adding linguistically-grounded phonology conditioning on top of the voting+fusion architecture would strengthen the contribution beyond just "LLM draft + gate + cross-attention."

**How to apply:** Wait for current runs to finish and analyze results first. Then discuss phonology integration design with user.
