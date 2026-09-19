"""Sign dictionary v0 (2026-09-19): gloss -> running-signing instances mined by the spotter.
Input  : spots tsv (clip_id, fr_a, fr_b, gloss, cos, margin) = spotted sub-sign segments.
Steps  : merge adjacent segments of one clip with the same gloss into ONE instance span (cos = max, n_seg = count),
         keep instances with cos >= --min_cos, rank per gloss by cos, keep the top --top_n.
Output : tsv  gloss, rank, clip_id, corpus, fr_a, fr_b, n_frames, n_seg, cos   + a stats json.
Also reports how much of the gloss-unit stream of a clip list (default pool_val) the dictionary covers (token and type level),
because lookup can only produce what has an entry; the rest falls back to embedding NN or fingerspelling.
  python align/build_sign_dictionary.py --out <tsv> [--min_cos 0.5 --top_n 50]
"""
import argparse, collections, glob, hashlib, json, os
CUR = '/home/rhong5/research_pro/hand_modeling_pro/asl_data_curation'


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--spots', default=f'{CUR}/data_records/alignment/spots_v4_anchor.tsv')
    ap.add_argument('--out', required=True); ap.add_argument('--min_cos', type=float, default=0.5); ap.add_argument('--top_n', type=int, default=50)
    ap.add_argument('--gloss_dir', default=f'{CUR}/data_records/alignment/gloss'); ap.add_argument('--index', default=f'{CUR}/data_records/curation/clip_index.tsv')
    ap.add_argument('--cover_list', default=f'{CUR}/data_records/curation/subsets/pool_val.txt')
    a = ap.parse_args()
    rows = []
    for ln in open(a.spots).read().split('\n')[1:]:
        if ln:
            c, fa, fb, g, cs, mg = ln.split('\t'); rows.append((c, int(fa), int(fb), g, float(cs)))
    rows.sort(key=lambda r: (r[0], r[1])); inst, cur = [], None
    for c, fa, fb, g, cs in rows:
        if cur and cur[0] == c and cur[3] == g and cur[2] == fa: cur = [c, cur[1], fb, g, max(cur[4], cs), cur[5] + 1]
        else:
            if cur: inst.append(tuple(cur))
            cur = [c, fa, fb, g, cs, 1]
    if cur: inst.append(tuple(cur))
    by = collections.defaultdict(list)
    for r in inst:
        if r[4] >= a.min_cos: by[r[3]].append(r)
    n_kept = 0
    with open(a.out, 'w') as f:
        f.write('gloss\trank\tclip_id\tcorpus\tfr_a\tfr_b\tn_frames\tn_seg\tcos\n')
        for g in sorted(by):
            for k, (c, fa, fb, _, cs, ns) in enumerate(sorted(by[g], key=lambda r: -r[4])[:a.top_n]):
                f.write(f'{g}\t{k}\t{c}\t{c.split(":")[0]}\t{fa}\t{fb}\t{fb - fa}\t{ns}\t{cs:.4f}\n'); n_kept += 1
    sizes = sorted(len(v) for v in by.values())
    st = {'spots': len(rows), 'instances_after_merge': len(inst), 'min_cos': a.min_cos, 'top_n': a.top_n, 'glosses': len(by), 'instances_kept': n_kept,
          'instances_per_gloss_before_topn': {'min': sizes[0], 'p25': sizes[len(sizes) // 4], 'median': sizes[len(sizes) // 2], 'p75': sizes[3 * len(sizes) // 4], 'max': sizes[-1]},
          'glosses_with_ge_5': sum(s >= 5 for s in sizes), 'glosses_with_ge_20': sum(s >= 20 for s in sizes),
          'mean_frames_per_instance': sum(r[2] - r[1] for v in by.values() for r in v) / max(1, sum(sizes))}
    # coverage of the gloss-unit stream of the clips in --cover_list
    want = set(open(a.cover_list).read().split()); text = {}
    hdr = None
    for ln in open(a.index):
        p = ln.rstrip('\n').split('\t')
        if hdr is None: hdr = p; ci = hdr.index('clip_id'); ti = hdr.index('text') if 'text' in hdr else hdr.index('caption'); continue
        if p[ci] in want: text[p[ci]] = p[ti].strip()
    g_of = {}
    for fp in glob.glob(os.path.join(a.gloss_dir, 'gloss_shard*.jsonl')):
        for ln in open(fp):
            d = json.loads(ln)
            if d.get('ok'): g_of[d['h']] = d['gloss']
    tok = hit = nclip = full = 0; types, thit = set(), set(); miss = collections.Counter()
    for c, t in text.items():
        gl = g_of.get(hashlib.md5(t.encode('utf-8')).hexdigest())
        if not gl: continue
        nclip += 1; h = [u.lower() in by for u in gl]; tok += len(gl); hit += sum(h); full += all(h)
        for u, ok in zip(gl, h):
            types.add(u.lower()); (thit.add(u.lower()) if ok else miss.update([u.lower()]))
    st['coverage'] = {'list': a.cover_list, 'clips_with_gloss': nclip, 'unit_tokens': tok, 'token_coverage': hit / max(1, tok), 'unit_types': len(types),
                      'type_coverage': len(thit) / max(1, len(types)), 'clips_fully_covered': full, 'top_missing': miss.most_common(40)}
    json.dump(st, open(a.out.replace('.tsv', '_stats.json'), 'w'), indent=1); print(json.dumps(st, indent=1))


if __name__ == '__main__':
    main()
