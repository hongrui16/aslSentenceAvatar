"""Score planner output against the locus test-set annotations (design doc §1.1 acceptance):
entity detection F1 (phrase.ref vs annotated refs), pronoun-target accuracy, verb (src,dst) accuracy,
locus consistency of the rule layer (pronoun phrase locus == locus assigned at first mention).
  python dialogue/score_plans.py --plans <jsonl>
"""
import argparse, json, re
from collections import Counter
def N(s):
    if isinstance(s, (list, tuple)): s = ' '.join(str(x) for x in s if x)
    if not isinstance(s, str): s = '' if s is None else str(s)
    return re.sub(r"[^a-z0-9 ]", '', s.lower().replace("'s", '')).strip()


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--plans', required=True); a = ap.parse_args()
    C = Counter(); per_cat = {}
    for l in open(a.plans):
        d = json.loads(l); first_locus = {}
        for t, p in zip(d['turns'], d['plans']):
            C['turns'] += 1; C['ok'] += p['ok']
            gold_refs = {N(k) for k in t.get('refs', {})}; pred_refs = {N(x['ref']) for x in p['phrases'] if x.get('ref')}
            C['ref_tp'] += len(gold_refs & pred_refs); C['ref_fp'] += len(pred_refs - gold_refs); C['ref_fn'] += len(gold_refs - pred_refs)
            for g in t.get('pron', []):
                C['pron'] += 1; w = g['word'].lower()
                hit = [x for x in p['phrases'] if w in re.sub(r"[^a-z' ]", ' ', x['text'].lower()).split() and x.get('pron')]
                if hit and N(hit[0].get('ref')) == N(g['ref']): C['pron_ok'] += 1
                if hit and hit[0].get('locus') and hit[0]['locus'] == first_locus.get(N(g['ref'])): C['locus_ok'] += 1
            for g in t.get('verbs', []):
                C['verb'] += 1; w = g['word'].lower()
                hit = [x for x in p['phrases'] if w in re.sub(r"[^a-z' ]", ' ', x['text'].lower()) and x.get('verb_dir')]
                if hit and [N(v) for v in hit[0]['verb_dir']] == [N(g['src']), N(g['dst'])]: C['verb_ok'] += 1
            for x in p['phrases']:
                if x.get('ref') and x.get('locus') and N(x['ref']) not in first_locus: first_locus[N(x['ref'])] = x['locus']
    P = C['ref_tp'] / max(C['ref_tp'] + C['ref_fp'], 1); R = C['ref_tp'] / max(C['ref_tp'] + C['ref_fn'], 1)
    res = {'turns': C['turns'], 'parse_ok': C['ok'] / max(C['turns'], 1), 'ref_P': P, 'ref_R': R, 'ref_F1': 2 * P * R / max(P + R, 1e-9),
           'pron_n': C['pron'], 'pron_acc': C['pron_ok'] / max(C['pron'], 1), 'locus_consistency': C['locus_ok'] / max(C['pron'], 1),
           'verb_n': C['verb'], 'verb_acc': C['verb_ok'] / max(C['verb'], 1)}
    print(json.dumps(res, indent=1))


if __name__ == '__main__': main()
