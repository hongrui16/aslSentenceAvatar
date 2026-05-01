"""
How2SignSMPLXPhonoDataset
=========================
Extends How2SignSMPLXDataset with a per-sentence pseudo-gloss string.

Pseudo-gloss extraction pipeline (same as tools/analyze_gloss_distribution.py):
    1. NLTK tokenize + POS tag
    2. Keep content POS (NOUN/VERB/ADJ/ADV)
    3. Rule-based lemmatize (tense + plural stripped)
    4. Drop ASL stopwords (be/do/have/modals)
    5. Join surviving lemmas with spaces → gloss_string

Extraction results are cached to disk (JSON keyed by sentence) so subsequent
training runs skip NLTK. Cache location: ``<project>/cache/pseudogloss_<mode>.json``
(override via ``cfg.PSEUDOGLOSS_CACHE_PATH``).

`__getitem__` returns (seq, sentence, gloss_string, length).
"""
import json
import os
import re
import sys
import nltk

if __name__ == "__main__" and __package__ is None:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataloader.How2SignSMPLXDataset import How2SignSMPLXDataset


PSEUDOGLOSS_CACHE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'cache'
)


for pkg in ['punkt', 'punkt_tab',
            'averaged_perceptron_tagger', 'averaged_perceptron_tagger_eng']:
    try:
        nltk.data.find(pkg)
    except LookupError:
        try: nltk.download(pkg, quiet=True)
        except Exception: pass


CONTENT_POS = {'N', 'V', 'J', 'R'}

ASL_STOPWORDS = {
    'be', 'is', 'am', 'are', 'was', 'were', 'being', 'been',
    'have', 'has', 'had', 'having',
    'do', 'does', 'did', 'done', 'doing',
    'will', 'would', 'shall', 'should',
    'can', 'could', 'may', 'might', 'must', 'ought',
    "'s", "'re", "'ve", "'d", "'ll", "'m", "n't",
    # Discourse / filler adverbs & connectives — low signal in sign language,
    # and `-ly` stripping mangles some of them (probably→probab).
    'so', 'therefore', 'thus', 'hence', 'however', 'moreover',
    'just', 'actually', 'probably', 'basically', 'simply',
    'probably', 'literally', 'essentially',
}

# Negations kept regardless of POS (NLTK often tags them DT/RB → normally dropped).
NEGATION_WORDS = {
    'no', 'not', 'never', 'nothing', 'none',
    'nobody', 'nowhere', 'neither', 'nor', 'cannot',
}

# Pronouns kept regardless of POS. NLTK tags them PRP/PRP$ which is outside
# CONTENT_POS, so they'd otherwise be dropped. In ASL they correspond to
# pointing signs (IX-1/IX-2/IX-3) and matter for conditioning.
PRONOUN_WORDS = {
    'i', 'you', 'we', 'he', 'she', 'they', 'it',
    'me', 'us', 'him', 'her', 'them',
    'my', 'your', 'our', 'his', 'their',
}

# WH-words kept regardless of POS (NLTK tags them WP/WDT/WRB — outside CONTENT_POS).
# All have dedicated ASL signs (WHAT, WHEN, WHERE, WHO, HOW, WHY) and carry
# either question structure or clause-linking meaning.
WH_WORDS = {
    'what', 'when', 'where', 'who', 'whom', 'whose', 'why', 'how', 'which',
}

# Auxiliary constructions to skip at the token level (handled in extract_gloss_string).
BE_FORMS = {"be", "am", "is", "are", "was", "were", "being", "been", "'s", "'re", "'m"}
HAVE_AUX = {"have", "has", "had", "having", "'ve", "'d"}  # "'s" too ambiguous

IRREGULAR_VERBS = {
    'is':'be','am':'be','are':'be','was':'be','were':'be','being':'be','been':'be',
    'has':'have','had':'have','having':'have',
    'does':'do','did':'do','done':'do','doing':'do',
    'went':'go','gone':'go','going':'go','goes':'go',
    'made':'make','making':'make','makes':'make',
    'ran':'run','running':'run','runs':'run',
    'saw':'see','seen':'see','seeing':'see','sees':'see',
    'took':'take','taken':'take','taking':'take','takes':'take',
    'got':'get','getting':'get','gets':'get','gotten':'get',
    'came':'come','coming':'come','comes':'come',
    'gave':'give','given':'give','giving':'give','gives':'give',
    'knew':'know','known':'know','knowing':'know','knows':'know',
    'thought':'think','thinking':'think','thinks':'think',
    'told':'tell','telling':'tell','tells':'tell',
    'said':'say','saying':'say','says':'say',
    'found':'find','finding':'find','finds':'find',
    # 'left':'leave' removed — collides with the direction "left" (e.g., "right and left").
    'leaving':'leave','leaves':'leave',
    'felt':'feel','feeling':'feel','feels':'feel',
    'brought':'bring','bringing':'bring','brings':'bring',
    'began':'begin','begun':'begin','beginning':'begin','begins':'begin',
    'kept':'keep','keeping':'keep','keeps':'keep',
    'held':'hold','holding':'hold','holds':'hold',
    'stood':'stand','standing':'stand','stands':'stand',
    'understood':'understand','understanding':'understand','understands':'understand',
    'wrote':'write','written':'write','writing':'write','writes':'write',
    'spoke':'speak','spoken':'speak','speaking':'speak','speaks':'speak',
    'read':'read','reading':'read','reads':'read',
    'broke':'break','broken':'break','breaking':'break','breaks':'break',
    'chose':'choose','chosen':'choose','choosing':'choose','chooses':'choose',
    'drew':'draw','drawn':'draw','drawing':'draw','draws':'draw',
    'drove':'drive','driven':'drive','driving':'drive','drives':'drive',
    'ate':'eat','eaten':'eat','eating':'eat','eats':'eat',
    'fell':'fall','fallen':'fall','falling':'fall','falls':'fall',
    'flew':'fly','flown':'fly','flying':'fly','flies':'fly',
    'forgot':'forget','forgotten':'forget','forgetting':'forget','forgets':'forget',
    'sang':'sing','sung':'sing','singing':'sing','sings':'sing',
    'swam':'swim','swum':'swim','swimming':'swim','swims':'swim',
    'threw':'throw','thrown':'throw','throwing':'throw','throws':'throw',
    'wore':'wear','worn':'wear','wearing':'wear','wears':'wear',
    'won':'win','winning':'win','wins':'win',
}

# Adjective / adverb irregular forms — checked when POS is J or R.
IRREGULAR_ADJ = {
    'best':'good','better':'good','worst':'bad','worse':'bad',
    'less':'little','lesser':'little','least':'little',
    'more':'many','most':'many',
    'further':'far','furthest':'far','farther':'far','farthest':'far',
    'elder':'old','eldest':'old',
}

# Plural-to-singular irregular nouns — checked when POS is N.
IRREGULAR_NOUNS = {
    'children':'child','feet':'foot','men':'man','women':'woman',
    'teeth':'tooth','mice':'mouse','people':'person','geese':'goose',
    'oxen':'ox',
}


def _restore_silent_e(base):
    """Heuristic: if base came from stripping -ing/-ed/-es and likely dropped a
    silent 'e' (moving→mov→move, using→us→use, writing→writ→write), add it back.

    Only fires when the base ends in a single vowel+consonant where the
    consonant isn't one of {w, x, y} (avoid fixing→fix→fixe, knowing→know→knowe).

    Guarded against common false positives where the stripped base is already
    a valid English word ending in -er / -en / -on (lowering→lower,
    covering→cover, happening→happen)."""
    if not base or base[-1] in 'aeiouy' or base[-1] in 'wx':
        return base
    # Skip when base is already a plausible word ending in a stressed/unstressed
    # -er/-en/-on syllable (length ≥ 4 avoids killing short verbs like fir→fire).
    if len(base) >= 4 and base[-2:] in ('er', 'en', 'on'):
        return base
    # Short cases: "us" (using), "ic" (icing) — V+C
    if len(base) == 2 and base[0] in 'aeiou':
        return base + 'e'
    # Longer cases: CVC ending preceded by consonant (writ, mov, mak, decid)
    if len(base) >= 3 and base[-2] in 'aeiou' and base[-3] not in 'aeiou':
        return base + 'e'
    return base


NATURAL_DOUBLES = {'ll', 'ss', 'ff', 'zz'}  # already-doubled stems, not inflectional


def _is_inflectional_double(base):
    """True if `base` ends in an inflectional doubled consonant (bigg, runn,
    shopp) rather than a naturally doubled stem (small, press, stuff)."""
    return (len(base) >= 2
            and base[-1] == base[-2]
            and base[-1] not in 'aeiouy'
            and base[-2:] not in NATURAL_DOUBLES)


def _strip_comp_suffix(w, suffix):
    """Strip a comparative/superlative suffix (-er / -est) and restore the
    base form, handling inflectional double-consonant (bigger→big) and silent-e
    (larger→large) patterns."""
    base = w[:-len(suffix)]
    if len(base) < 2:
        return w
    if _is_inflectional_double(base):
        return base[:-1]
    return _restore_silent_e(base)


def _simple_lemmatize(w, pos):
    # POS-gated irregular tables.
    if pos == 'V' and w in IRREGULAR_VERBS: return IRREGULAR_VERBS[w]
    if pos in ('J', 'R') and w in IRREGULAR_ADJ: return IRREGULAR_ADJ[w]
    if pos == 'N' and w in IRREGULAR_NOUNS:     return IRREGULAR_NOUNS[w]
    if len(w) <= 3: return w
    if pos == 'V':
        if w.endswith('ying') and len(w) > 4: return w[:-4] + 'y'
        if w.endswith('ing'):
            base = w[:-3]
            if len(base) < 2: return w
            if _is_inflectional_double(base):
                return base[:-1]
            return _restore_silent_e(base)
        if w.endswith('ied'): return w[:-3] + 'y'
        if w.endswith('ed'):
            base = w[:-2]
            if len(base) < 2: return w
            if _is_inflectional_double(base):
                return base[:-1]
            return _restore_silent_e(base)
        if w.endswith('es') and len(w) > 4:
            base = w[:-2]
            # Only strip 'es' when stem ends in sibilant (box→boxes, miss→misses,
            # watch→watches). Otherwise "moves"→"mov" is wrong; fall through to
            # the -s rule instead.
            if base.endswith(('s', 'sh', 'ch', 'x', 'z')):
                return base
        if w.endswith('s') and not w.endswith('ss') and len(w) > 3: return w[:-1]
    elif pos == 'N':
        if w.endswith('ies') and len(w) > 4: return w[:-3] + 'y'
        if w.endswith('es') and len(w) > 4:
            base = w[:-2]
            if base.endswith(('s', 'sh', 'ch', 'x', 'z')):
                return base
        if w.endswith('s') and not w.endswith('ss') and len(w) > 3: return w[:-1]
    elif pos in ('J', 'R'):
        # -ier / -iest: happier→happy, easiest→easy
        if w.endswith('iest') and len(w) > 4: return w[:-4] + 'y'
        if w.endswith('ier')  and len(w) > 4: return w[:-3] + 'y'
        # -est / -er with doubled-consonant + silent-e: bigger→big, largest→large.
        # Applies to both adjectives (JJR/JJS) and gradable adverbs (RBR/RBS).
        if w.endswith('est') and len(w) > 4: return _strip_comp_suffix(w, 'est')
        if w.endswith('er')  and len(w) > 3: return _strip_comp_suffix(w, 'er')
        if pos == 'R' and w.endswith('ly') and len(w) > 4: return w[:-2]
    return w


def _prev_word(lows, tags_v, i, allowed_skip_tags=('RB',)):
    """Walk back from position i past tokens whose POS starts with any tag in
    `allowed_skip_tags` (default: adverbs). Returns (prev_index, prev_word) or
    (None, None) if the scan falls off the start."""
    j = i - 1
    while j >= 0:
        tag = tags_v[j][1] or ''
        if any(tag.startswith(t) for t in allowed_skip_tags):
            j -= 1
            continue
        return j, lows[j]
    return None, None


def _build_drop_mask(toks):
    """Return set of token indices to drop as auxiliary constructions.

    Patterns handled (optional intervening adverbs allowed):
      - (be-form) [RB*] + going + to + VB*    → drop "going" (future auxiliary)
      - (have-aux) [RB*] + got                → drop "got"    (got as aux "have got")
      - got + to + VB*                        → drop "got"    (modal "got to")
    """
    drop = set()
    lows = [t.lower() for t in toks]
    tags_v = nltk.pos_tag(toks)
    for i, tl in enumerate(lows):
        if tl == 'going':
            _, prev = _prev_word(lows, tags_v, i)
            if (prev in BE_FORMS
                    and i + 2 < len(lows) and lows[i+1] == 'to'
                    and tags_v[i+2][1].startswith('VB')):
                drop.add(i)
        elif tl == 'got':
            _, prev = _prev_word(lows, tags_v, i)
            if prev in HAVE_AUX:
                drop.add(i)
            elif (i + 2 < len(lows) and lows[i+1] == 'to'
                  and tags_v[i+2][1].startswith('VB')):
                drop.add(i)
    return drop


def extract_gloss_string(sentence: str) -> str:
    """Extract content-word lemma sequence, drop ASL stopwords, lowercase, space-joined."""
    toks = nltk.word_tokenize(sentence)
    tags = nltk.pos_tag(toks)
    drop_idx = _build_drop_mask(toks)
    lemmas = []
    for i, (tok, tag) in enumerate(tags):
        if i in drop_idx:
            continue
        raw = tok.lower()
        # NLTK splits "don't"/"won't"/"can't" → ["do"/"wo"/"ca", "n't"]; the "n't"
        # token carries the negation and is normalized to "not" here.
        if raw == "n't":
            lemmas.append('not')
            continue
        # Preserve negations even though NLTK tags them DT/RB (e.g., "no", "not").
        if raw in NEGATION_WORDS:
            lemmas.append(raw)
            continue
        # Preserve pronouns (PRP/PRP$ — outside CONTENT_POS). ASL indexes them.
        if raw in PRONOUN_WORDS:
            lemmas.append(raw)
            continue
        # Preserve WH-words (WP/WDT/WRB) — question/clause signs in ASL.
        if raw in WH_WORDS:
            lemmas.append(raw)
            continue
        if not tag or tag[0] not in CONTENT_POS:
            continue
        t = re.sub(r"[^a-zA-Z']", '', tok).lower()
        if not t or t in ASL_STOPWORDS:
            continue
        l = _simple_lemmatize(t, tag[0])
        if l in ASL_STOPWORDS:
            continue
        lemmas.append(l)
    return ' '.join(lemmas)


class How2SignSMPLXPhonoDataset(How2SignSMPLXDataset):
    """Same as How2SignSMPLXDataset but `__getitem__` emits gloss_string
    in place of the (previously duplicated) sentence slot."""

    def __init__(self, mode='train', cfg=None, logger=None):
        super().__init__(mode=mode, cfg=cfg, logger=logger)
        self._gloss_strings = self._load_or_build_gloss_cache(mode)

    def _load_or_build_gloss_cache(self, mode):
        cache_path = getattr(self.cfg, 'PSEUDOGLOSS_CACHE_PATH', None) if self.cfg else None
        if cache_path is None:
            os.makedirs(PSEUDOGLOSS_CACHE_DIR, exist_ok=True)
            cache_path = os.path.join(PSEUDOGLOSS_CACHE_DIR, f'pseudogloss_{mode}.json')

        cache = {}
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'r', encoding='utf-8') as f:
                    cache = json.load(f)
            except Exception as e:
                if self.logger is not None:
                    self.logger.warning(
                        f"[{mode}] failed to load pseudogloss cache {cache_path} ({e}); recomputing"
                    )
                cache = {}

        gloss_strings = []
        n_new = 0
        for sentence, _ in self.data_list:
            g = cache.get(sentence)
            if g is None:
                g = extract_gloss_string(sentence)
                cache[sentence] = g
                n_new += 1
            gloss_strings.append(g)

        if n_new > 0:
            tmp_path = cache_path + '.tmp'
            try:
                with open(tmp_path, 'w', encoding='utf-8') as f:
                    json.dump(cache, f, ensure_ascii=False, indent=2)
                os.replace(tmp_path, cache_path)
            except Exception as e:
                if self.logger is not None:
                    self.logger.warning(f"[{mode}] failed to save pseudogloss cache: {e}")

        if self.logger is not None:
            example = gloss_strings[0] if gloss_strings else ''
            if n_new == 0:
                self.logger.info(
                    f"[{mode}] loaded {len(gloss_strings)} gloss strings from cache "
                    f"{cache_path} (example: '{example}')"
                )
            else:
                self.logger.info(
                    f"[{mode}] cached {len(gloss_strings)} gloss strings at {cache_path} "
                    f"({n_new} newly computed; example: '{example}')"
                )
        return gloss_strings

    def __getitem__(self, idx):
        result = super().__getitem__(idx)
        if self.use_fk_cache:
            seq, sentence, _, length, gt_joints44 = result
            gloss_string = self._gloss_strings[idx] if length > 0 else ''
            return seq, sentence, gloss_string, length, gt_joints44
        seq, sentence, _, length = result
        gloss_string = self._gloss_strings[idx] if length > 0 else ''
        return seq, sentence, gloss_string, length


if __name__ == "__main__":
    """Pre-build pseudo-gloss caches for all splits.

    Usage:
        python -m dataloader.How2SignSMPLXPhonoDataset
        python -m dataloader.How2SignSMPLXPhonoDataset --modes train val
        python -m dataloader.How2SignSMPLXPhonoDataset --root_dir /scratch/.../Neural-Sign-Actors
    """
    import argparse
    import logging

    parser = argparse.ArgumentParser()
    parser.add_argument("--modes", nargs='+', default=['train', 'val', 'test'],
                        choices=['train', 'val', 'test'])
    parser.add_argument("--root_dir", type=str,
                        default='/scratch/rhong5/dataset/Neural-Sign-Actors',
                        help="Dataset root containing how2sign_realigned_*.xlsx and {mode}_poses/")
    parser.add_argument("--cache_path", type=str, default=None,
                        help="Override cache path (default: ./cache/pseudogloss_{mode}.json)")
    parser.add_argument("--target_len", type=int, default=200)
    parser.add_argument("--force", action="store_true",
                        help="Delete existing cache file before rebuilding")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    logger = logging.getLogger("pseudogloss_cache")

    # Only sentences are touched here — pose loading/format flags (USE_ROT6D,
    # USE_UPPER_BODY) are irrelevant for gloss extraction and left at defaults.
    class _Cfg:
        ROOT_DIR = args.root_dir
        TARGET_SEQ_LEN = args.target_len
        CAMERA = 'rgb_front'
        PSEUDOGLOSS_CACHE_PATH = args.cache_path  # None → default per-mode path

    os.makedirs(PSEUDOGLOSS_CACHE_DIR, exist_ok=True)

    for mode in args.modes:
        cache_file = args.cache_path or os.path.join(
            PSEUDOGLOSS_CACHE_DIR, f'pseudogloss_{mode}.json'
        )
        if args.force and os.path.exists(cache_file):
            logger.info(f"[{mode}] --force: removing existing cache {cache_file}")
            os.remove(cache_file)

        logger.info(f"==== building pseudogloss cache for '{mode}' ====")
        ds = How2SignSMPLXPhonoDataset(mode=mode, cfg=_Cfg(), logger=logger)

        non_empty = [g for g in ds._gloss_strings if g]
        logger.info(
            f"[{mode}] {len(ds._gloss_strings)} samples, "
            f"{len(non_empty)} non-empty glosses, "
            f"avg_len={sum(len(g.split()) for g in non_empty) / max(1, len(non_empty)):.2f} tokens"
        )
        for i in range(min(3, len(ds.data_list))):
            sent, _ = ds.data_list[i]
            logger.info(f"  [{i}] sentence: {sent[:80]}")
            logger.info(f"      gloss   : {ds._gloss_strings[i]}")

    logger.info("done.")
