---
name: Pseudo-Gloss Extraction Pipeline
description: How2SignSMPLXPhonoDataset builds pseudo-gloss strings (English sentence → content-word lemma sequence) with disk cache and ASL-specific lexical rules
type: project
---

`dataloader/How2SignSMPLXPhonoDataset.py` converts How2Sign English captions to space-joined pseudo-gloss strings for use as gloss-conditioning input.

**Disk cache:**
- Location: `./cache/pseudogloss_{train,val,test}.json` (override via `cfg.PSEUDOGLOSS_CACHE_PATH`)
- Keyed by **sentence string** (dict), not index — robust to xlsx reordering / incremental additions
- Atomic write via `tmp → os.replace`; incremental: computes only missing sentences
- Rebuild all splits: `python -m dataloader.How2SignSMPLXPhonoDataset --force`
- Rebuild one split: `... --force --modes test`
- The `__main__` block only touches sentences, so rot6d/upper_body flags are irrelevant there; `CAMERA='rgb_front'` must stay consistent with training to match the filtered xlsx row set.

**Non-obvious extraction rules (all in `extract_gloss_string`):**

1. **Pronouns preserved** (`i/you/we/he/she/they/it/my/your/...`). NLTK tags them PRP/PRP$ which is outside CONTENT_POS {N,V,J,R} so they'd otherwise be dropped. ASL indexes them (IX-1/IX-2/IX-3) so keeping them is linguistically faithful and helps conditioning.

2. **Negations preserved** (`no/not/never/nothing/none/nobody/...`). Same reason — tagged DT/RB, otherwise dropped. Critical for sentence meaning.

3. **`n't` normalized to `not`.** NLTK splits `don't/won't/can't/doesn't/didn't/haven't/shouldn't` → `["do"/"wo"/"ca"/..., "n't"]`. Without this the negation is silently lost.

4. **Discourse markers dropped** (added to ASL_STOPWORDS): `so/therefore/thus/hence/however/moreover/just/actually/probably/basically/simply/literally/essentially`. Low signal in sign; also `probably` was getting mangled into `probab` by the `-ly` rule.

5. **Auxiliary bigram drops** (handled in `_build_drop_mask`):
   - `(be-form) + going + to + VB*` → drop `going` (future auxiliary). `going to + NOUN` keeps `going` as motion verb.
   - `(have-aux) + got` → drop `got` ("have got" = have)
   - `got + to + VB*` → drop `got` (modal "got to" = have to)

6. **POS-gated irregular tables.** Split into `IRREGULAR_VERBS` (V only), `IRREGULAR_ADJ` (J/R only — best/better/worst/worse/more/less/further), `IRREGULAR_NOUNS` (N only — children/feet/men/teeth/mice/people). The un-gated version caused `leaves` (plural of leaf) to map to `leave` regardless of context. Also explicitly removed `left:leave` — it collides with the direction "left".

7. **Natural doubled consonants preserved.** `NATURAL_DOUBLES = {'ll','ss','ff','zz'}` — don't treat these as inflectional doubling. Fixes `smaller → small` (not `smal`), `pressing → press`, `dressed → dress`, `calling → call`. Other doubles (bigger→big, running→run, hottest→hot) still de-doubled.

8. **Silent-e restoration heuristic** (`_restore_silent_e`) for `moving → move`, `using → use`, `writing → write`, `deciding → decide`, `firing → fire`. Guarded against `-er/-en/-on` endings with len ≥ 4 (avoids `lowering → lowere`, `covering → covere`, `happening → happene`).

**Known remaining tail cases** (NLTK tagger limitations, no dict available — not worth chasing):
- `largest → larg` (VCCe silent-e: would need `-rg/-ng` rule, but that breaks `hanging → hange`)
- `hotter/safer/wider/taller` sometimes tagged NN in isolation → comparative rule doesn't fire

**Why:** Pseudo-gloss is the condition signal for `--cond_mode={gloss, sentence_gloss}` experiments. Quality matters — we iterated through ~8 bug classes with user feedback on real How2Sign outputs.

**How to apply:** If gloss extraction quality issues surface (negation lost, weird lemmatization), investigate `_simple_lemmatize` / `extract_gloss_string` first, then rebuild the cache with `--force`. Don't manually edit the JSON cache.
