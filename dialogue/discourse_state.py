"""Step 5 planning layer, rule part: cross-turn spatial discourse state (referent -> locus) + NMM rules.

Loci are shared world positions (4 slots). The LLM only names referents; slot assignment is deterministic here so
the behaviour is testable without a model. Positions are offsets in shoulder widths from the signer's chest,
(x right+, z forward+); the retrieval/stitch layer applies them to the citation-position segments.
"""
import re

SLOTS = {'L1': (0.35, 0.15), 'L2': (-0.35, 0.15), 'L3': (0.55, 0.35), 'L4': (-0.55, 0.35)}
WH = re.compile(r"^\s*(who|what|where|when|why|how|which|whose)\b", re.I)
NEG = re.compile(r"\b(not|n't|never|no|nothing|nobody|none)\b", re.I)


def normalize_ref(r):
    if isinstance(r, (list, tuple)): r = ' '.join(str(x) for x in r if x)  # LLM sometimes emits a list
    if not isinstance(r, str): r = '' if r is None else str(r)
    return re.sub(r"[^a-z0-9 ]", '', r.lower().strip().replace("'s", '')).strip()


class DiscourseState:
    def __init__(self, n_slots=4):
        self.slots = list(SLOTS)[:n_slots]; self.ref2loc = {}; self.clock = 0; self.last_use = {}
    def _touch(self, ref): self.clock += 1; self.last_use[ref] = self.clock
    def assign(self, ref):
        """First mention (or re-mention) -> locus; evicts the least recently used referent when full."""
        ref = normalize_ref(ref)
        if not ref: return None
        if ref in self.ref2loc: self._touch(ref); return self.ref2loc[ref]
        used = set(self.ref2loc.values()); free = [s for s in self.slots if s not in used]
        if free: loc = free[0]
        else:
            victim = min(self.ref2loc, key=lambda r: self.last_use[r]); loc = self.ref2loc.pop(victim)
        self.ref2loc[ref] = loc; self._touch(ref); return loc
    def refer(self, ref):
        """Anaphoric reference: returns the locus only if the referent is already established."""
        ref = normalize_ref(ref)
        if ref in self.ref2loc: self._touch(ref); return self.ref2loc[ref]
        return None
    def snapshot(self): return dict(self.ref2loc)


def sentence_type(text):
    t = text.strip()
    if t.endswith('?'): return 'wh' if WH.search(t) else 'yn'
    return 'decl'


def nmm_for(text, role=None):
    """Non-manual markers from sentence type + negation + topic role (rule-based)."""
    st = sentence_type(text); out = []
    if st == 'wh': out.append('brow_down')
    if st == 'yn': out.append('brow_up')
    if NEG.search(text): out.append('headshake')
    if role == 'topic': out.append('topic_brow')
    return out


def apply_plan(state, turn_text, phrases):
    """phrases: list of dicts from the LLM {text, role, ref, pron, verb_dir, fs}. Fills locus / dir / nmm using the state.
    Returns the completed phrase list and the state update made in this turn."""
    before = state.snapshot(); sent_nmm = nmm_for(turn_text)
    PRON = {'he', 'she', 'they', 'him', 'her', 'them', 'his', 'hers', 'their', 'theirs', 'it', 'its'}
    def most_recent():  # rule fallback for unresolved pronouns: most recently used referent
        return max(state.ref2loc, key=lambda r: state.last_use[r]) if state.ref2loc else None
    for p in phrases:
        ref = p.get('ref'); p['locus'] = None; p['dir'] = None
        words = set(re.sub(r"[^a-z' ]", ' ', str(p.get('text', '')).lower()).split())
        if (p.get('pron') or (words & PRON and not ref)) and (not ref or normalize_ref(ref) in PRON):
            ref = most_recent(); p['ref'] = ref; p['pron'] = True; p['ref_source'] = 'rule_recent'
        vd = p.get('verb_dir')
        if isinstance(vd, list) and len(vd) == 2:
            vd = [most_recent() if normalize_ref(v) in PRON else v for v in vd]; p['verb_dir'] = vd
        if ref and not p.get('pron'): p['locus'] = state.assign(ref)
        elif ref and p.get('pron'): p['locus'] = state.refer(ref) or state.assign(ref)
        vd = p.get('verb_dir')
        if isinstance(vd, dict): vd = [vd.get('src') or vd.get('source'), vd.get('dst') or vd.get('target')]
        if vd and isinstance(vd, list) and len(vd) == 2 and all(isinstance(v, (str, list)) and v for v in vd):
            role = lambda v: 'SELF' if normalize_ref(v) in ('i', 'me', 'my', 'self', 'speaker') else 'ADDR' if normalize_ref(v) in ('you', 'your', 'addressee', 'listener') else None
            src = state.refer(vd[0]) or role(vd[0]); dst = state.refer(vd[1]) or role(vd[1])
            p['dir'] = [src, dst]
        p['nmm'] = sorted(set(sent_nmm + (['topic_brow'] if p.get('role') == 'topic' else [])))
    after = state.snapshot(); upd = {k: v for k, v in after.items() if before.get(k) != v}
    return phrases, upd
