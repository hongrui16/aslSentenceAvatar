"""Pseudo-gloss keywords for every unique pooled caption (user 09-17: chunks keep tense/plural/function words; use the
llm_draft_gloss recipe instead). Same prompt as tools/generate_llm_draft_gloss.py (prompts/pseudogloss_extraction_prompt.txt):
drop function words / auxiliaries / fillers, keep one unit per sign, lemmatize to citation form. Output jsonl per shard:
{"h": md5(text), "text": ..., "gloss": [...], "ok": 1}; ok=0 -> rule fallback (stopword removal + crude lemma).
Backend: vLLM on a full GPU, transformers on a MIG slice (dialogue/llm_backend.py). Resumable.
  python align/extract_gloss_pooled.py --out_dir <dir> --shard 0 --nshards 4
"""
import argparse, csv, hashlib, json, os, re, sys
_repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, _repo)
from dialogue.llm_backend import Chat
INDEX = '/home/rhong5/research_pro/hand_modeling_pro/asl_data_curation/data_records/curation/clip_index.tsv'
PROMPT = os.path.join(_repo, 'prompts', 'pseudogloss_extraction_prompt.txt')
STOP = set("a an the to of in on at for with and or but is are was were be been am it its this that i you he she we they me him her us them my your his our their do does did have has had will would can could should not no so if then there here what who how when where why which".split())
norm = lambda s: re.sub(r"[^a-z0-9' ]+", ' ', s.lower()).split()


def fallback(t):
    out = []
    for w in norm(t):
        if w in STOP: continue
        for suf in ('ing', 'ed', 'es', 's'):
            if w.endswith(suf) and len(w) - len(suf) >= 3: w = w[:-len(suf)]; break
        out.append(w)
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--index', default=INDEX); ap.add_argument('--out_dir', required=True)
    ap.add_argument('--model', default='Qwen/Qwen2.5-14B-Instruct'); ap.add_argument('--shard', type=int, default=0); ap.add_argument('--nshards', type=int, default=1)
    ap.add_argument('--max_items', type=int, default=0); ap.add_argument('--batch', type=int, default=1000); ap.add_argument('--hf_batch', type=int, default=48)
    a = ap.parse_args(); os.makedirs(a.out_dir, exist_ok=True)
    tmpl = open(PROMPT, encoding='utf-8').read()
    texts = {}
    with open(a.index) as f:
        for r in csv.DictReader(f, delimiter='\t', quoting=csv.QUOTE_NONE):
            t = r['text'].strip()
            if t: texts.setdefault(hashlib.md5(t.encode()).hexdigest(), t)
    items = sorted(texts.items())[a.shard::a.nshards]
    out = os.path.join(a.out_dir, f'gloss_shard{a.shard}.jsonl'); done = set()
    if os.path.exists(out):
        with open(out) as f: done = {json.loads(l)['h'] for l in f if l.strip()}
    items = [(h, t) for h, t in items if h not in done and len(t) < 600]
    if a.max_items: items = items[:a.max_items]
    print(f'shard {a.shard}: {len(items)} to gloss ({len(done)} done)', flush=True)
    if not items: print(a.shard, 'DONE 0'); return
    llm = Chat(a.model, 4096, 0.9); n, n_ok = 0, 0
    with open(out, 'a') as fo:
        for i in range(0, len(items), a.batch):
            chunk = items[i:i + a.batch]
            msgs = [[{'role': 'user', 'content': tmpl.replace('{input_sentence}', t)}] for _, t in chunk]
            outs = llm(msgs, temperature=0.0, max_tokens=96, batch=a.hf_batch)
            for (h, t), raw in zip(chunk, outs):
                g = raw.strip().split('\n')[0].strip().lower(); g = re.sub(r'^gloss:\s*', '', g); words = [w for w in re.sub(r"[^a-z0-9' \-]+", ' ', g).split() if w]
                ok = int(bool(words) and len(words) <= 2 * max(1, len(norm(t))) and all(len(w) <= 25 for w in words))
                if not ok: words = fallback(t)
                fo.write(json.dumps({'h': h, 'text': t, 'gloss': words, 'ok': ok}, ensure_ascii=False) + '\n'); n += 1; n_ok += ok
            fo.flush(); print(f'{a.shard} {n}/{len(items)} ok {n_ok/max(n,1):.3f}', flush=True)
    print(a.shard, 'DONE', n, f'ok_rate {n_ok/max(n,1):.3f}', flush=True)


if __name__ == '__main__': main()
