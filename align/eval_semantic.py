"""Semantic end metric for retrieval-and-stitch (2026-09-18): does the stitched sequence carry the same SIGNS as the GT clip?
Whole-clip DTW measures motion/style similarity (motion-NN oracle 0.68, every text-side improvement ~1.0), so it cannot judge the text side.
Here both the GT clip and the retrieved segments are labelled by ONE FIXED spotter (the v4-anchor aligner's motion embedding, nearest
SignBank gloss text among 2,820, cos >= --min_cos), and we compare gloss multisets:
  recall    = |GT glosses ∩ retrieved glosses| / |GT glosses|      precision = ... / |retrieved glosses|
  random baseline = same number of random bank segments, labelled the same way.  lift = recall / recall_random.
GT labels = val_spots tsv (fixed spotter + caption-hit filter, so GT glosses are trustworthy); retrieved-segment labels = fixed spotter
without the caption filter (the retrieved segment has no caption of its own).
  python align/eval_semantic.py --align_ckpt <pt> --bank_dir <dir> --units chunks|gloss --out <json>
"""
import argparse, csv, glob, hashlib, json, os, random, sys
import numpy as np, torch, torch.nn.functional as F
_repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, _repo)
from align.align_dataset import INDEX, merge_stopword_chunks
from align.align_dataset_v2 import load_chunks
from align.train_align_v4 import SignBank, DR
from align.spot_signs import norm, STOP

ANCHOR = '/scratch/rhong5/weights/temp_training_weights/aslSentenceAvatar/Align_Pooled/20260916_133241_job315245_v4_anchor/best_model.pt'
ANCHOR_BANK = '/projects/kosecka/hongrui/dataset/smplx_fits/pooled_tokens/seg_bank_v4_anchor'


def load_bank(d, dev):
    emb, meta, cids, off = [], [], [], 0
    for fp in sorted(glob.glob(os.path.join(d, 'bank_*.npz'))):
        z = np.load(fp); emb.append(z['emb']); m = z['meta'].copy(); m[:, 0] += off; meta.append(m); cids.extend([str(x) for x in z['clip_ids']]); off = len(cids)
    E = torch.from_numpy(np.concatenate(emb).astype(np.float16)).to(dev)
    for i in range(0, len(E), 1_000_000): E[i:i + 1_000_000] = F.normalize(E[i:i + 1_000_000].float(), dim=-1).half()
    return E, np.concatenate(meta), cids


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--align_ckpt', required=True); ap.add_argument('--bank_dir', required=True)
    ap.add_argument('--units', choices=['chunks', 'gloss'], required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--val_list', default='/home/rhong5/research_pro/hand_modeling_pro/asl_data_curation/data_records/curation/subsets/pool_val.txt')
    ap.add_argument('--val_spots', default=f'{DR}/val_spots_v4_anchor.tsv'); ap.add_argument('--n_clips', type=int, default=400); ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--min_cos', type=float, default=0.4); ap.add_argument('--judge_ckpt', default=ANCHOR); ap.add_argument('--judge_bank', default=ANCHOR_BANK); ap.add_argument('--judge_sb', default=f'{DR}/signbank_tokens.npz')
    a = ap.parse_args(); random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed); dev = 'cuda'
    rows = {}
    with open(INDEX) as f:
        for r in csv.DictReader(f, delimiter='\t', quoting=csv.QUOTE_NONE): rows[r['clip_id']] = (r['video'], r['npz'], r['text'].strip(), r['corpus'])
    wanted = [l.strip() for l in open(a.val_list) if l.strip()]; wanted = [c for c in wanted if c in rows and rows[c][2]]
    random.shuffle(wanted); wanted = wanted[:a.n_clips]  # same 400 clips as eval_retrieval_v2 (same seed/order)
    gt = {}
    for r in csv.DictReader(open(a.val_spots), delimiter='\t'): gt.setdefault(r['clip_id'], []).append(r['gloss'])
    # retrieval model + its bank
    from align.train_align_v6 import AlignModel
    ck = torch.load(a.align_ckpt, map_location='cpu'); c = ck['config']
    model = AlignModel(c['mmm_ckpt'], c['text_encoder'], c['e']).to(dev); model.load_state_dict(ck['model']); model.eval()
    E, meta, cids = load_bank(a.bank_dir, dev); seg_video = np.array([rows[cids[i]][0] if cids[i] in rows else '?' for i in meta[:, 0]])
    # fixed spotter: anchor model's bank embeddings (same segment order as any bank built from pretrain_all) + gloss vectors
    ck2 = torch.load(a.judge_ckpt, map_location='cpu'); c2 = ck2['config']
    anchor = AlignModel(c2['mmm_ckpt'], c2['text_encoder'], c2['e']).to(dev); anchor.load_state_dict(ck2['model']); anchor.eval()
    EA, metaA, cidsA = load_bank(a.judge_bank, dev); assert len(EA) == len(E) and cidsA == cids, 'banks must share segment order'
    sb = SignBank(a.judge_sb, f'{DR}/signbank_split.json', anchor.mmm.pos.num_embeddings)
    Z, texts = [], [sb.texts[i] for i in range(len(sb.stems))]
    with torch.no_grad():
        for i in range(0, len(sb.stems), 128):
            x, m, spans, _ = sb.batch(list(range(i, min(i + 128, len(sb.stems)))))
            with torch.autocast('cuda', dtype=torch.bfloat16): Z.append(anchor.embed_segments(x.to(dev), m.to(dev), spans).float())
    Z = torch.cat(Z); uniq = sorted(set(texts)); ti = {t: k for k, t in enumerate(uniq)}
    G = torch.zeros(len(uniq), Z.shape[1], device=dev); cnt = torch.zeros(len(uniq), device=dev)
    for z, t in zip(Z, texts): G[ti[t]] += z; cnt[ti[t]] += 1
    G = F.normalize(G / cnt[:, None], dim=-1)
    def label(seg_idx):
        sim = EA[seg_idx].float() @ G.T; cos, g = sim.max(1); return [uniq[int(k)] if float(cv) >= a.min_cos else None for cv, k in zip(cos, g)]
    h2c = load_chunks(f'{DR}/{a.units}'); rng = np.random.default_rng(a.seed)
    rec, prec, rec_r, prec_r, n_gt, n_used = [], [], [], [], [], 0; per_corpus = {}; n_ret, n_ret_lab = [], []
    for cid in wanted:
        video, npz, text, co = rows[cid]; g_gt = gt.get(cid)
        if not g_gt: continue
        u = h2c.get(hashlib.md5(text.encode()).hexdigest()); u = (u if h2c.get('__gloss__') else merge_stopword_chunks(u)) if u else [text]
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16): zt = model.embed_chunks(u, dev).half()
        sim = (zt @ E.T).float(); sim[:, torch.from_numpy(seg_video == video).to(dev)] = -1e4; top = sim.argmax(-1)
        rnd = torch.from_numpy(rng.choice(len(E), size=len(top))).to(dev)
        lab, lab_r = [g for g in label(top) if g], [g for g in label(rnd) if g]; n_ret.append(len(top)); n_ret_lab.append(len(lab))
        S = set(g_gt); R, RR = set(lab), set(lab_r)
        rec.append(len(S & R) / len(S)); prec.append(len(S & R) / max(len(R), 1)); rec_r.append(len(S & RR) / len(S)); prec_r.append(len(S & RR) / max(len(RR), 1))
        n_gt.append(len(S)); n_used += 1; per_corpus.setdefault(co, []).append(rec[-1])
    out = {'n_clips_with_gt_glosses': n_used, 'mean_gt_glosses': float(np.mean(n_gt)), 'mean_retrieved': float(np.mean(n_ret)), 'mean_retrieved_labelled': float(np.mean(n_ret_lab)), 'units': a.units, 'bank_dir': a.bank_dir, 'val_spots': a.val_spots, 'judge_ckpt': a.judge_ckpt,
           'recall': float(np.mean(rec)), 'precision': float(np.mean(prec)), 'recall_random': float(np.mean(rec_r)), 'precision_random': float(np.mean(prec_r)),
           'lift_recall': float(np.mean(rec) / max(np.mean(rec_r), 1e-9)), 'per_corpus_recall': {k: float(np.mean(v)) for k, v in per_corpus.items()}}
    print(json.dumps(out, indent=1)); os.makedirs(os.path.dirname(a.out) or '.', exist_ok=True); json.dump(out, open(a.out, 'w'), indent=1)


if __name__ == '__main__': main()
