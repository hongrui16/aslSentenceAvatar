"""Step 5 locus test set, candidate generation (design doc §3): LLM writes short 2-speaker dialogues of three
categories with entity/pronoun/verb annotations; a validator keeps only well-formed ones. Human curation later.

Categories: ix (pronoun back-reference to an entity introduced earlier), agr (directional/agreement verbs with
explicit source and target), multi (>= 2 entities in play, alternating reference).
Output jsonl: {"id", "cat", "turns":[{"speaker","text","refs":{"canonical":"mention"}, "pron":[{"word","ref"}],
"verbs":[{"word","src","dst"}]}]}

  python dialogue/make_locus_testset.py --out <jsonl> --n_per_cat 170
"""
import argparse, json, os, random, re, sys
_repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, _repo)
SYS = ('You write short natural English dialogues between two people, A and B, for testing sign language generation. '
       'Return ONLY a JSON object: {"turns":[{"speaker":"A"|"B","text":"...","refs":{"<canonical entity name>":"<the words in this turn that mention it>"},'
       '"pron":[{"word":"<pronoun in this turn>","ref":"<canonical entity it refers to>"}],'
       '"verbs":[{"word":"<directional verb in this turn>","src":"<canonical entity or I/you>","dst":"<canonical entity or I/you>"}]}]}. '
       'Entities are third-person people or things (never A or B themselves). Use canonical names consistently across turns '
       '(e.g. "mother", "john", "the car" -> "car"). Every pronoun (he, she, it, they, him, her, them, his, hers) that refers to an entity '
       'must be listed in "pron". Directional verbs are: give, tell, ask, help, show, send, pay, inform, meet, teach, visit, call. '
       'Turns are 4 to 14 words, 3 to 6 turns, plain everyday topics, no names of real celebrities.')
CATS = {
 'ix': 'Category: pronoun back-reference. Turn 1 introduces one entity by name or noun; at least two later turns refer to it only with pronouns. Exactly one entity in the dialogue.',
 'agr': 'Category: directional verbs. Introduce one or two entities; at least two turns use directional verbs whose source and target are entities or I/you (e.g. "I told her", "he gave it to my brother").',
 'multi': 'Category: multiple referents. Introduce two or three entities in the first turns; later turns alternate references between them, using pronouns at least twice and a directional verb at least once.'}


def valid(d, cat):
    try:
        T = d['turns']; assert 3 <= len(T) <= 6
        ents = set(); npron = 0; nverb = 0
        for t in T:
            assert t['speaker'] in ('A', 'B') and 3 <= len(t['text'].split()) <= 16
            for c, m in t.get('refs', {}).items():
                assert m.lower().replace('.', '') in t['text'].lower(); ents.add(c.lower())
            for p in t.get('pron', []): assert p['word'].lower() in t['text'].lower().split() or p['word'].lower() in re.sub(r"[^a-z' ]", ' ', t['text'].lower()).split(); assert p['ref'].lower() in ents; npron += 1
            for v in t.get('verbs', []): assert v['word'].lower() in re.sub(r"[^a-z' ]", ' ', t['text'].lower()); nverb += 1
        if cat == 'ix': return len(ents) == 1 and npron >= 2
        if cat == 'agr': return 1 <= len(ents) <= 2 and nverb >= 2
        return len(ents) >= 2 and npron >= 2 and nverb >= 1
    except Exception: return False


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--out', required=True); ap.add_argument('--n_per_cat', type=int, default=170)
    ap.add_argument('--model', default='Qwen/Qwen2.5-14B-Instruct'); ap.add_argument('--seed', type=int, default=0); ap.add_argument('--gpu_mem', type=float, default=0.9)
    a = ap.parse_args(); rng = random.Random(a.seed)
    topics = ['family', 'school', 'work', 'shopping', 'travel', 'cooking', 'sports', 'pets', 'neighbors', 'weekend plans', 'doctor visit', 'birthday', 'moving house', 'a broken phone', 'a job interview', 'a lost wallet']
    from dialogue.llm_backend import Chat
    llm = Chat(a.model, 2048, a.gpu_mem)
    reqs = [(cat, rng.choice(topics), i) for cat in CATS for i in range(int(a.n_per_cat * 2.5))]  # oversample, validator filters
    msgs = [[{'role': 'system', 'content': SYS}, {'role': 'user', 'content': f"{CATS[cat]} Topic: {tp}. Variation {i}."}] for cat, tp, i in reqs]
    outs = llm(msgs, temperature=0.9, top_p=0.95, max_tokens=900, seed=a.seed); kept = {c: [] for c in CATS}; n_parse = 0
    for (cat, tp, i), o in zip(reqs, outs):
        m = re.search(r'\{.*\}', o, re.S)
        if not m: continue
        try: d = json.loads(m.group(0)); n_parse += 1
        except Exception: continue
        if valid(d, cat) and len(kept[cat]) < a.n_per_cat: kept[cat].append({'id': f'{cat}_{len(kept[cat]):03d}', 'cat': cat, 'topic': tp, 'turns': d['turns']})
    with open(a.out, 'w') as fo:
        for cat in CATS:
            for d in kept[cat]: fo.write(json.dumps(d, ensure_ascii=False) + '\n')
    print('generated', len(reqs), 'parsed', n_parse, 'kept', {c: len(v) for c, v in kept.items()}, '->', a.out, flush=True)


if __name__ == '__main__': main()
