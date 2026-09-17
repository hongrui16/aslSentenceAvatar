"""v2 (2026-09-17): load_chunks also accepts gloss_*.jsonl (key "gloss") so text units can be pseudo-gloss keywords. v1 = align_dataset.py (untouched).
Clip-level dataset for roadmap 4.2 contrastive alignment: per clip = VQ token streams (from tokenize_corpus.py),
non-rest segment spans in TOKEN index space (from segment_clips.py boundaries, frame/4), and phrase-chunk strings
(from chunk_phrases.py jsonl; clips whose text hash is missing fall back to the whole sentence as one chunk).
"""
import csv, hashlib, json, os, random
import numpy as np, torch
from torch.utils.data import Dataset

PARTS = ['body', 'lhand', 'rhand', 'face']
INDEX = '/home/rhong5/research_pro/hand_modeling_pro/asl_data_curation/data_records/curation/clip_index.tsv'
SEGMENTS = '/home/rhong5/research_pro/hand_modeling_pro/asl_data_curation/data_records/alignment/segments/segments_all.tsv'


STOP = set("a an the to of in on at for with and or but is are was were be been am it its this that i you he she we they "
           "my your his her our their me him her us them do does did have has had will would can could should".split())


def merge_stopword_chunks(chunks):
    """Merge blocks made only of function words into the next block (previous if last) — Qwen sometimes emits them alone."""
    out = []
    carry = ''
    for c in chunks:
        words = [w.strip(".,!?;:'\"").lower() for w in c.split()]
        if words and all(w in STOP or not w for w in words): carry = (carry + ' ' + c).strip()
        else: out.append((carry + ' ' + c).strip()); carry = ''
    if carry:
        if out: out[-1] = (out[-1] + ' ' + carry).strip()
        else: out.append(carry)
    return out or chunks


def load_chunks(chunk_dir):
    """Text units per caption hash. A dir of chunks_*.jsonl gives phrase blocks (key 'chunks'); a dir of gloss_*.jsonl
    gives pseudo-gloss keywords (key 'gloss', one unit per sign, lemmatized, function words dropped)."""
    h2c = {}
    if chunk_dir and os.path.isdir(chunk_dir):
        for fn in sorted(os.listdir(chunk_dir)):
            if fn.endswith('.jsonl'):
                with open(os.path.join(chunk_dir, fn)) as f:
                    for l in f:
                        if l.strip():
                            r = json.loads(l); u = r.get('chunks') or r.get('gloss')
                            if u: h2c[r['h']] = u
    return h2c


class ClipAlignDataset(Dataset):
    def __init__(self, tok_dir, list_txt, chunk_dir=None, index_tsv=INDEX, segments_tsv=SEGMENTS,
                 max_tok=256, max_chunks=30, max_segs=60, train=True, max_items=None):
        wanted = [l.strip() for l in open(list_txt) if l.strip()]; wset = set(wanted)
        text = {}
        with open(index_tsv) as f:
            for r in csv.DictReader(f, delimiter='\t', quoting=csv.QUOTE_NONE):
                if r['clip_id'] in wset: text[r['clip_id']] = r['text'].strip()
        segs = {}
        with open(segments_tsv) as f:
            next(f)
            for l in f:
                cid, T, fps, bounds, rest = l.rstrip('\n').split('\t')
                if cid in wset:
                    b = list(map(int, bounds.split(','))); rf = list(map(int, rest.split(',')))
                    segs[cid] = [(a, c) for a, c, r in zip(b[:-1], b[1:], rf) if not r]
        h2c = load_chunks(chunk_dir); self.items = []; n_fallback = 0
        for cid in wanted:
            t = text.get(cid, ''); ss = segs.get(cid, [])
            tok = os.path.join(tok_dir, cid.replace(':', '__') + '.npz')
            if not t or not ss or not os.path.exists(tok): continue
            ch = h2c.get(hashlib.md5(t.encode()).hexdigest())
            if not ch: ch = [t]; n_fallback += 1
            else: ch = merge_stopword_chunks(ch)
            self.items.append((tok, ch[:max_chunks], ss[:max_segs]))
            if max_items and len(self.items) >= max_items: break
        self.max_tok, self.train = max_tok, train
        print(f'[ClipAlignDataset] {len(self.items)} clips ({n_fallback} whole-sentence fallback)', flush=True)

    def __len__(self): return len(self.items)

    def __getitem__(self, i):
        tok, chunks, segs = self.items[i]
        d = np.load(tok); x = np.stack([d[p].astype(np.int64) for p in PARTS], 1)  # (n_tok, 4)
        spans = [(a // 4, max(a // 4 + 1, -(-b // 4))) for a, b in segs]
        n = len(x)
        if n > self.max_tok:  # crop a token window; keep segments fully inside, need >= 1
            s = random.randint(0, n - self.max_tok) if self.train else (n - self.max_tok) // 2
            x = x[s:s + self.max_tok]
            spans = [(a - s, min(b - s, self.max_tok)) for a, b in spans if a >= s and a < s + self.max_tok]
            if not spans: spans = [(0, len(x))]
        spans = [(a, min(b, len(x))) for a, b in spans]
        return torch.from_numpy(x), spans, chunks


def collate(batch):
    L = max(len(b[0]) for b in batch)
    x = torch.zeros(len(batch), L, 4, dtype=torch.long); m = torch.zeros(len(batch), L, dtype=torch.bool)
    spans, chunks, seg_own, chk_own = [], [], [], []
    for i, (xi, sp, ch) in enumerate(batch):
        x[i, :len(xi)] = xi; m[i, :len(xi)] = True
        for a, b in sp: spans.append((i, a, b)); seg_own.append(i)
        for c in ch: chunks.append(c); chk_own.append(i)
    return x, m, spans, torch.tensor(seg_own), chunks, torch.tensor(chk_own)
