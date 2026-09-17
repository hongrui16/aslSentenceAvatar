"""Step 5 planning layer: dialogue history -> sign plan per turn (local Qwen via vLLM + rule-based discourse state).

Input jsonl: one dialogue per line {"id":..., "turns":[{"speaker":"A","text":"..."}, ...]} (locus test set format).
Output jsonl: same + "plans": per turn {"phrases":[{text, role, ref, pron, verb_dir, fs, locus, dir, nmm}], "sent_type",
"state_update", "ok"} where ok=1 iff the LLM JSON parsed and its phrases cover the turn's words exactly.

  python dialogue/plan_dialogue.py --dialogues <jsonl> --out <jsonl> [--max_items N]
"""
import argparse, json, os, re, sys
_repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, _repo)
from dialogue.discourse_state import DiscourseState, apply_plan, sentence_type

SYS = ('You are a planner for American Sign Language production. Given a dialogue history and the CURRENT turn, '
       'split the current turn into contiguous phrase blocks in the original word order (1-4 words each; every word '
       'of the turn appears in exactly one block). For each block give: "role" (topic|subj|verb|obj|other), '
       '"ref" = the entity the block denotes if it is a person/thing that can be placed in signing space, as a short '
       'canonical name reused consistently across turns (e.g. "mother", "john", "car"), else null; "pron" = true if '
       'the block is a pronoun or anaphoric reference to an entity already mentioned earlier in the dialogue; '
       '"verb_dir" = [source, target] canonical names (or "I"/"you") if the block is a directional verb like give, tell, '
       'ask, help, show, send, pay, inform, meet, else null; "fs" = true if the block must be fingerspelled (proper '
       'noun, brand, unusual name). RULES: a block with "pron": true MUST carry "ref" = the canonical name of the entity it '
       'refers to, resolved from the dialogue history (he/she/they/him/her/them/his/their -> the person or thing mentioned before); '
       '"verb_dir" entries must be canonical names or "I"/"you", never pronouns; only true directional verbs get "verb_dir" '
       '(not see, notice, come, go, be). Reply with ONLY a JSON array of block objects.')
EX_HIST = 'A: Where does your mother live?\nCURRENT TURN (B): My mother lives in Fairfax.'
EX_OUT = '[{"text":"My mother","role":"topic","ref":"mother","pron":false,"verb_dir":null,"fs":false},' \
         '{"text":"lives","role":"verb","ref":null,"pron":false,"verb_dir":null,"fs":false},' \
         '{"text":"in Fairfax.","role":"obj","ref":"fairfax","pron":false,"verb_dir":null,"fs":true}]'
EX2_HIST = 'A: My mother lives in Fairfax.\nB: Does she like it there?\nCURRENT TURN (A): Yes, and she told me to visit her soon.'
EX2_OUT = '[{"text":"Yes, and","role":"other","ref":null,"pron":false,"verb_dir":null,"fs":false},' \
          '{"text":"she","role":"subj","ref":"mother","pron":true,"verb_dir":null,"fs":false},' \
          '{"text":"told me","role":"verb","ref":null,"pron":false,"verb_dir":["mother","I"],"fs":false},' \
          '{"text":"to visit","role":"verb","ref":null,"pron":false,"verb_dir":["I","mother"],"fs":false},' \
          '{"text":"her soon.","role":"obj","ref":"mother","pron":true,"verb_dir":null,"fs":false}]'
norm = lambda s: re.sub(r"[^a-z0-9' ]+", ' ', s.lower()).split()


def build_prompt(turns, k):
    hist = '\n'.join(f"{t['speaker']}: {t['text']}" for t in turns[:k]) or '(dialogue start)'
    return f"{hist}\nCURRENT TURN ({turns[k]['speaker']}): {turns[k]['text']}"


def parse(raw, text):
    m = re.search(r'\[.*\]', raw, re.S)
    if not m: return None
    try: cand = json.loads(m.group(0))
    except Exception: return None
    if not (isinstance(cand, list) and cand and all(isinstance(c, dict) and c.get('text') for c in cand)): return None
    if norm(' '.join(c['text'] for c in cand)) != norm(text): return None
    return cand


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--dialogues', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--model', default='Qwen/Qwen2.5-14B-Instruct'); ap.add_argument('--max_items', type=int, default=0); ap.add_argument('--gpu_mem', type=float, default=0.9)
    a = ap.parse_args()
    dlgs = [json.loads(l) for l in open(a.dialogues) if l.strip()]
    if a.max_items: dlgs = dlgs[:a.max_items]
    from dialogue.llm_backend import Chat
    llm = Chat(a.model, 2048, a.gpu_mem)
    reqs = [(di, k) for di, d in enumerate(dlgs) for k in range(len(d['turns']))]
    msgs = [[{'role': 'system', 'content': SYS}, {'role': 'user', 'content': EX_HIST}, {'role': 'assistant', 'content': EX_OUT},
             {'role': 'user', 'content': EX2_HIST}, {'role': 'assistant', 'content': EX2_OUT},
             {'role': 'user', 'content': build_prompt(dlgs[di]['turns'], k)}] for di, k in reqs]
    outs = llm(msgs, temperature=0.0, max_tokens=768); raw = {r: o for r, o in zip(reqs, outs)}
    n_ok = 0
    with open(a.out, 'w') as fo:
        for di, d in enumerate(dlgs):
            st = DiscourseState(); plans = []
            for k, t in enumerate(d['turns']):
                cand = parse(raw[(di, k)], t['text']); ok = int(cand is not None)
                phrases = cand or [{'text': t['text'], 'role': 'other', 'ref': None, 'pron': False, 'verb_dir': None, 'fs': False}]
                phrases, upd = apply_plan(st, t['text'], phrases)
                plans.append({'phrases': phrases, 'sent_type': sentence_type(t['text']), 'state_update': upd, 'state': st.snapshot(), 'ok': ok}); n_ok += ok
            d['plans'] = plans; fo.write(json.dumps(d, ensure_ascii=False) + '\n')
    print(f'planned {len(reqs)} turns in {len(dlgs)} dialogues, parse+coverage ok {n_ok/len(reqs):.3f} -> {a.out}', flush=True)


if __name__ == '__main__': main()
