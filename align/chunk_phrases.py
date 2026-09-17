"""Qwen phrase chunking for roadmap 4.2 (text side): split each unique caption into contiguous phrase blocks
("my mother" / "lives" / "in Fairfax") with a local Qwen via vLLM. Output jsonl per shard:
{"h": md5(text), "text": ..., "chunks": [...], "ok": 1}   (ok=0 -> parse/coverage failed, chunks = [whole sentence]).

  python align/chunk_phrases.py --out_dir <dir> [--shard 0 --nshards 1] [--max_items 50]
"""
import argparse, csv, hashlib, json, os, re, sys

INDEX = '/home/rhong5/research_pro/hand_modeling_pro/asl_data_curation/data_records/curation/clip_index.tsv'
SYS = ('You segment English sentences into short contiguous phrase blocks for sign language alignment. '
       'Rules: keep the original words and their order; each block is 1-4 words and a meaningful unit '
       '(noun phrase, verb group, prepositional phrase); every word of the sentence appears in exactly one block; '
       'reply with ONLY a JSON array of strings, no explanation.')
EX_IN = 'My mother lives in Fairfax.'
EX_OUT = '["My mother", "lives", "in Fairfax."]'

norm = lambda s: re.sub(r"[^a-z0-9' ]+", ' ', s.lower()).split()


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--index', default=INDEX); ap.add_argument('--out_dir', required=True)
    ap.add_argument('--model', default='Qwen/Qwen2.5-14B-Instruct'); ap.add_argument('--shard', type=int, default=0); ap.add_argument('--nshards', type=int, default=1)
    ap.add_argument('--max_items', type=int, default=0); ap.add_argument('--batch', type=int, default=2000); ap.add_argument('--gpu_mem', type=float, default=0.92)
    a = ap.parse_args(); os.makedirs(a.out_dir, exist_ok=True)
    texts = {}
    with open(a.index) as f:
        for r in csv.DictReader(f, delimiter='\t', quoting=csv.QUOTE_NONE):
            t = r['text'].strip()
            if t: texts.setdefault(hashlib.md5(t.encode()).hexdigest(), t)
    items = sorted(texts.items())[a.shard::a.nshards]
    out = os.path.join(a.out_dir, f'chunks_shard{a.shard}.jsonl'); done = set()
    if os.path.exists(out):
        with open(out) as f: done = {json.loads(l)['h'] for l in f if l.strip()}
    items = [(h, t) for h, t in items if h not in done and len(t) < 600]
    if a.max_items: items = items[:a.max_items]
    print(f'shard {a.shard}: {len(items)} to chunk ({len(done)} already done)', flush=True)
    if not items: print(a.shard, 'DONE 0', flush=True); return
    from vllm import LLM, SamplingParams
    llm = LLM(model=a.model, dtype='bfloat16', max_model_len=1024, gpu_memory_utilization=a.gpu_mem, enforce_eager=False)
    sp = SamplingParams(temperature=0.0, max_tokens=512)
    n, n_ok = 0, 0
    with open(out, 'a') as fo:
        for i in range(0, len(items), a.batch):
            chunk = items[i:i + a.batch]
            msgs = [[{'role': 'system', 'content': SYS}, {'role': 'user', 'content': EX_IN}, {'role': 'assistant', 'content': EX_OUT}, {'role': 'user', 'content': t}] for _, t in chunk]
            outs = llm.chat(msgs, sp, use_tqdm=False)
            for (h, t), o in zip(chunk, outs):
                raw = o.outputs[0].text.strip(); ok = 0; blocks = [t]
                m = re.search(r'\[.*\]', raw, re.S)
                if m:
                    try:
                        cand = json.loads(m.group(0))
                        if isinstance(cand, list) and cand and all(isinstance(c, str) and c.strip() for c in cand):
                            if norm(' '.join(cand)) == norm(t): blocks, ok = [c.strip() for c in cand], 1
                    except Exception: pass
                fo.write(json.dumps({'h': h, 'text': t, 'chunks': blocks, 'ok': ok}, ensure_ascii=False) + '\n'); n += 1; n_ok += ok
            fo.flush(); print(f'{a.shard} {n}/{len(items)} ok {n_ok/max(n,1):.3f}', flush=True)
    print(a.shard, 'DONE', n, f'ok_rate {n_ok/max(n,1):.3f}', flush=True)


if __name__ == '__main__': main()
