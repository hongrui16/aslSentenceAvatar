"""
Analyze per-sentence gloss matching distribution for How2Sign.

Pipeline per sentence:
    1. Tokenize + POS tag
    2. Keep content POS (NOUN, VERB, ADJ, ADV, PROPN-optional)
    3. Lemmatize
    4. Lookup in SignBank / ASL-LEX vocabularies
    5. Count matched glosses

Output: distribution stats + per-bucket counts.
"""
import re
import pandas as pd
import numpy as np
from collections import Counter

import nltk

for pkg in ['punkt', 'punkt_tab', 'averaged_perceptron_tagger',
            'averaged_perceptron_tagger_eng']:
    try: nltk.data.find(pkg)
    except LookupError:
        try: nltk.download(pkg, quiet=True)
        except Exception: pass


# Rule-based lemmatizer — same approach as readme.md preprocessing
IRREGULAR_VERBS = {
    # copula: be
    'is':'be','am':'be','are':'be','was':'be','were':'be','being':'be','been':'be',
    # aux: have
    'has':'have','had':'have','having':'have',
    # aux: do
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
    'left':'leave','leaving':'leave','leaves':'leave',
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
    'gave':'give','sang':'sing','sung':'sing','singing':'sing','sings':'sing',
    'swam':'swim','swum':'swim','swimming':'swim','swims':'swim',
    'threw':'throw','thrown':'throw','throwing':'throw','throws':'throw',
    'wore':'wear','worn':'wear','wearing':'wear','wears':'wear',
    'won':'win','winning':'win','wins':'win',
    'best':'good','better':'good','worst':'bad','worse':'bad',
    'children':'child','feet':'foot','men':'man','women':'woman',
    'teeth':'tooth','mice':'mouse','people':'person',
}

def simple_lemmatize(w, pos):
    """Strip common inflectional suffixes. Mirrors readme preprocessing."""
    if w in IRREGULAR_VERBS: return IRREGULAR_VERBS[w]
    if len(w) <= 3: return w
    if pos == 'V':  # verb
        if w.endswith('ying') and len(w) > 4: return w[:-4] + 'y'
        if w.endswith('ing'):
            base = w[:-3]
            if len(base) >= 2 and base[-1] == base[-2]: base = base[:-1]  # running → run
            return base if len(base) >= 2 else w
        if w.endswith('ied'): return w[:-3] + 'y'
        if w.endswith('ed'):
            base = w[:-2]
            if len(base) >= 2 and base[-1] == base[-2]: base = base[:-1]
            return base if len(base) >= 2 else w
        if w.endswith('es') and len(w) > 4: return w[:-2]
        if w.endswith('s') and not w.endswith('ss') and len(w) > 3: return w[:-1]
    elif pos == 'N':  # noun
        if w.endswith('ies') and len(w) > 4: return w[:-3] + 'y'
        if w.endswith('es') and len(w) > 4: return w[:-2]
        if w.endswith('s') and not w.endswith('ss') and len(w) > 3: return w[:-1]
    elif pos == 'J':  # adj
        if w.endswith('iest'): return w[:-4] + 'y'
        if w.endswith('ier'):  return w[:-3] + 'y'
        if w.endswith('est') and len(w) > 4: return w[:-3]
        if w.endswith('er')  and len(w) > 3: return w[:-2]
    elif pos == 'R':  # adverb
        if w.endswith('ly') and len(w) > 4: return w[:-2]
    return w


ROOT = '/home/rhong5/research_pro/hand_modeling_pro/aslSentenceAvatar'
SIGNBANK_CSV = f'{ROOT}/data/ASL_signbank/asl_signbank_dictionary-export.csv'
ASLLEX_CSV   = f'{ROOT}/data/ASL_LEX2.0/ASL-LEX_View_Data.csv'
H2S_XLSX     = '/scratch/rhong5/dataset/Neural-Sign-Actors/how2sign_realigned_train.xlsx'

CONTENT_POS = {'N', 'V', 'J', 'R'}  # NN*, VB*, JJ*, RB*

# ASL-specific stopwords — words that are not signed (copula/aux/modal),
# even though they are technically content POS in English.
ASL_STOPWORDS = {
    'be', 'is', 'am', 'are', 'was', 'were', 'being', 'been',
    'have', 'has', 'had', 'having',
    'do', 'does', 'did', 'done', 'doing',
    'will', 'would', 'shall', 'should',
    'can', 'could', 'may', 'might', 'must', 'ought',
    "'s", "'re", "'ve", "'d", "'ll", "'m", "n't",
}


def build_vocab():
    """Return dict: lemma_lower -> source tag ('SB' | 'AL' | 'both')."""
    vocab = {}

    # SignBank: Lemma ID Gloss, Annotation ID Gloss, Keywords
    df = pd.read_csv(SIGNBANK_CSV)
    added = 0
    for _, row in df.iterrows():
        toks = []
        for col in ['Lemma ID Gloss', 'Annotation ID Gloss']:
            v = str(row.get(col, '')).strip()
            if v and v.lower() != 'nan':
                toks.append(v.replace('-', ' ').replace('_', ' '))
        kw = str(row.get('Keywords', ''))
        if kw and kw.lower() != 'nan':
            toks += [k.strip() for k in kw.split(',')]
        for t in toks:
            for w in t.lower().split():
                w = re.sub(r"[^a-z']", '', w)
                if w and w not in vocab:
                    vocab[w] = 'SB'; added += 1
    print(f'[SignBank] {added} unique tokens added, total vocab = {len(vocab)}')

    # ASL-LEX 2.0: Entry ID / Lemma ID (e.g., "1_dollar" -> "dollar")
    df2 = pd.read_csv(ASLLEX_CSV)
    added = 0
    for col in ['Entry ID', 'Lemma ID']:
        if col in df2.columns:
            for v in df2[col].dropna():
                v = str(v).strip().strip('"').lower()
                for w in re.split(r'[_\s\-]+', v):
                    w = re.sub(r"[^a-z']", '', w)
                    if w:
                        if w not in vocab:
                            vocab[w] = 'AL'; added += 1
                        elif vocab[w] == 'SB':
                            vocab[w] = 'both'
    print(f'[ASL-LEX]  {added} new tokens, total vocab = {len(vocab)}')
    return vocab


def extract_matched_glosses(sent, vocab):
    toks = nltk.word_tokenize(sent)
    tags = nltk.pos_tag(toks)
    matched = []
    content = 0
    for tok, tag in tags:
        if not tag or tag[0] not in CONTENT_POS:
            continue
        t = re.sub(r"[^a-zA-Z']", '', tok).lower()
        if not t or t in ASL_STOPWORDS:
            continue
        content += 1
        l = simple_lemmatize(t, tag[0])
        if l in ASL_STOPWORDS:
            continue
        if l in vocab or t in vocab:
            matched.append(l if l in vocab else t)
    return matched, content


def main():
    vocab = build_vocab()
    df = pd.read_excel(H2S_XLSX)
    sentences = [str(s).strip() for s in df['SENTENCE'].dropna()]
    print(f'[How2Sign] {len(sentences)} sentences')

    matched_counts, content_counts = [], []
    all_matched = Counter()

    for i, s in enumerate(sentences):
        m, c = extract_matched_glosses(s, vocab)
        matched_counts.append(len(m))
        content_counts.append(c)
        all_matched.update(m)
        if (i+1) % 5000 == 0:
            print(f'  processed {i+1}')

    mc = np.array(matched_counts)
    cc = np.array(content_counts)

    print('\n' + '='*60)
    print('Per-sentence matched-gloss distribution')
    print('='*60)
    print(f'  mean       : {mc.mean():.2f}')
    print(f'  median     : {np.median(mc):.0f}')
    print(f'  P25 / P75  : {np.percentile(mc,25):.0f} / {np.percentile(mc,75):.0f}')
    print(f'  std        : {mc.std():.2f}')
    print(f'  min / max  : {mc.min()} / {mc.max()}')
    print()
    for thr, label in [(0,'==0'), (1,'<=1'), (2,'<=2'), (3,'<=3'),
                        (4,'>=4'), (6,'>=6'), (10,'>=10')]:
        if label.startswith('<=') or label.startswith('=='):
            n = (mc <= thr).sum() if label.startswith('<=') else (mc == thr).sum()
        else:
            n = (mc >= thr).sum()
        print(f'  sentences with N {label:<5}: {n:6d} ({100*n/len(mc):5.2f}%)')

    print('\n' + '='*60)
    print('Per-sentence content-word distribution (pre-lookup)')
    print('='*60)
    print(f'  mean={cc.mean():.2f}  median={np.median(cc):.0f}  '
          f'P25={np.percentile(cc,25):.0f}  P75={np.percentile(cc,75):.0f}')

    print('\n' + '='*60)
    print('Match rate (token-level)')
    print('='*60)
    tot_c = cc.sum(); tot_m = mc.sum()
    print(f'  content tokens : {tot_c}')
    print(f'  matched tokens : {tot_m}  ({100*tot_m/tot_c:.1f}%)')

    print('\n' + '='*60)
    print('Top-20 matched glosses')
    print('='*60)
    for w, n in all_matched.most_common(20):
        print(f'  {w:<15} {n}')


if __name__ == '__main__':
    main()
