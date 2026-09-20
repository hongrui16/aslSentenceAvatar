"""Sign dictionary v2 (2026-09-20; v1 = build_sign_dictionary.py untouched): gloss -> running-signing instances from the OPEN-VOCABULARY
miner (mine_open_vocab.py -> refine_open_vocab_v2.py), optionally united with the citation-NN spots (spot_signs_v2.py).
Why v2: v1 ranks by the spotter cosine and only knows the ~1,500 glosses that have a SignBank citation video (unit coverage 0.47).
The open-vocabulary candidates cover 7,581 gloss units and carry a confidence that was validated against ground truth with two
independent judges: precision is monotone in `proto` (>= 0.5 about 50 %, >= 0.6 about 63 %, >= 0.7 about 73 %).
Steps: merge adjacent same-gloss segments of a clip into one instance (proto = max over the merged segments); keep instances with
proto >= --min_proto; drop words whose prototype rests on fewer than --min_proto_vid videos or that keep fewer than --min_inst
instances; per word keep the --top_n best with at most --per_video per video (signer diversity for unit selection later).
Output: tsv gloss rank clip_id corpus video fr_a fr_b n_frames n_seg proto source + stats json with the coverage of the gloss-unit
stream of --cover_list (token level, type level, fully covered clips), for the open-vocabulary entries alone and for the union.
  python align/build_sign_dictionary_v2.py --cands <..._refined_v2.tsv> --out <tsv> [--citation <spots_v2_anchor.tsv>] [--min_proto 0.5]
"""
import argparse, collections, csv, glob, hashlib, json, os
CUR = '/home/rhong5/research_pro/hand_modeling_pro/asl_data_curation'


def merge(rows):  # rows: (clip, fa, fb, gloss, conf) -> instances (clip, fa, fb, gloss, conf_max, n_seg)
    rows.sort(key=lambda r: (r[0], r[3], r[1])); inst, cur = [], None
    for c, fa, fb, g, cf in rows:
        if cur and cur[0] == c and cur[3] == g and fa <= cur[2]: cur = [c, cur[1], max(cur[2], fb), g, max(cur[4], cf), cur[5] + 1]
        else:
            if cur: inst.append(tuple(cur))
            cur = [c, fa, fb, g, cf, 1]
    if cur: inst.append(tuple(cur))
    return inst


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--cands', required=True); ap.add_argument('--out', required=True); ap.add_argument('--citation', default='')
    ap.add_argument('--min_proto', type=float, default=0.5); ap.add_argument('--min_proto_vid', type=int, default=3); ap.add_argument('--min_inst', type=int, default=3)
    ap.add_argument('--top_n', type=int, default=50); ap.add_argument('--per_video', type=int, default=5); ap.add_argument('--citation_min_cos', type=float, default=0.5)
    ap.add_argument('--gloss_dir', default=f'{CUR}/data_records/alignment/gloss'); ap.add_argument('--index', default=f'{CUR}/data_records/curation/clip_index.tsv')
    ap.add_argument('--cover_list', default=f'{CUR}/data_records/curation/subsets/pool_val.txt')
    a = ap.parse_args()
    text, vid_of = {}, {}; want = set(open(a.cover_list).read().split())
    with open(a.index) as f:
        for r in csv.DictReader(f, delimiter='\t', quoting=csv.QUOTE_NONE):
            vid_of[r['clip_id']] = r['corpus'] + ':' + r['video']
            if r['clip_id'] in want: text[r['clip_id']] = r['text'].strip()
    lines = open(a.cands).read().split('\n'); col = {h: i for i, h in enumerate(lines[0].split('\t'))}; rows = []; pvid = {}
    for ln in lines[1:]:
        if not ln: continue
        p = ln.split('\t'); g = p[col['gloss']]; pvid[g] = int(p[col['proto_vid']])
        if float(p[col['proto']]) >= a.min_proto: rows.append((p[col['clip_id']], int(p[col['fr_a']]), int(p[col['fr_b']]), g, float(p[col['proto']])))
    n_words_in = len(pvid); by = collections.defaultdict(list)
    for r in merge(rows):
        if pvid[r[3]] >= a.min_proto_vid: by[r[3]].append(r + ('openvocab',))
    by = {g: v for g, v in by.items() if len(v) >= a.min_inst}; ov_words = set(by)
    if a.citation:
        crows = []
        for ln in open(a.citation).read().split('\n')[1:]:
            if ln:
                c, fa, fb, g, cs, mg = ln.split('\t')
                if float(cs) >= a.citation_min_cos: crows.append((c, int(fa), int(fb), g, float(cs)))
        for r in merge(crows): by.setdefault(r[3], []).append(r + ('citation',))
    n_kept = 0; per_word_n = []
    with open(a.out, 'w') as f:
        f.write('gloss\trank\tclip_id\tcorpus\tvideo\tfr_a\tfr_b\tn_frames\tn_seg\tconf\tsource\n')
        for g in sorted(by):
            cnt = collections.Counter(); k = 0
            for c, fa, fb, _, cf, ns, src in sorted(by[g], key=lambda r: (r[6] != 'openvocab', -r[4])):  # open-vocab entries first (validated confidence), then citation
                v = vid_of.get(c, c)
                if cnt[v] >= a.per_video or k >= a.top_n: continue
                cnt[v] += 1; f.write(f'{g}\t{k}\t{c}\t{c.split(":")[0]}\t{v}\t{fa}\t{fb}\t{fb - fa}\t{ns}\t{cf:.4f}\t{src}\n'); k += 1
            n_kept += k; per_word_n.append(k)
    g_of = {}
    for fp in glob.glob(os.path.join(a.gloss_dir, 'gloss_shard*.jsonl')):
        for ln in open(fp):
            d = json.loads(ln)
            if d.get('ok'): g_of[d['h']] = [u.lower().strip() for u in d['gloss'] if u.strip()]

    def coverage(vocab):
        tok = hit = nclip = full = 0; types, thit = set(), set(); miss = collections.Counter()
        for c, t in text.items():
            gl = g_of.get(hashlib.md5(t.encode('utf-8')).hexdigest())
            if not gl: continue
            nclip += 1; h = [u in vocab for u in gl]; tok += len(gl); hit += sum(h); full += all(h)
            for u, ok in zip(gl, h): types.add(u); (thit.add(u) if ok else miss.update([u]))
        return {'clips_with_gloss': nclip, 'unit_tokens': tok, 'token_coverage': hit / max(1, tok), 'type_coverage': len(thit) / max(1, len(types)),
                'clips_fully_covered': full, 'top_missing': miss.most_common(30)}
    pw = sorted(per_word_n)
    st = {'cands': a.cands, 'citation': a.citation, 'min_proto': a.min_proto, 'min_proto_vid': a.min_proto_vid, 'min_inst': a.min_inst, 'top_n': a.top_n, 'per_video': a.per_video,
          'words_in_candidates': n_words_in, 'words_openvocab': len(ov_words), 'words_total': len(by), 'instances_kept': n_kept,
          'instances_per_word': {'p25': pw[len(pw) // 4], 'median': pw[len(pw) // 2], 'p75': pw[3 * len(pw) // 4]}, 'words_with_ge_10': sum(x >= 10 for x in pw),
          'coverage_openvocab': coverage(ov_words), 'coverage_union': coverage(set(by))}
    json.dump(st, open(a.out.replace('.tsv', '_stats.json'), 'w'), indent=1)
    for k in ['coverage_openvocab', 'coverage_union']: st[k] = {kk: vv for kk, vv in st[k].items() if kk != 'top_missing'} | {'top_missing': [w for w, _ in st[k]['top_missing'][:20]]}
    print(json.dumps(st, indent=1))


if __name__ == '__main__':
    main()
