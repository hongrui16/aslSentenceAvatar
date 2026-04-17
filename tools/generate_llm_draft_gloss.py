"""
Generate LLM draft pseudo-glosses using a local open-source model.

Runs Qwen2.5-32B-Instruct (default) locally on GPU to extract pseudo-gloss
from each sentence in the How2Sign dataset. Results are cached to
``./cache/llm_draft_gloss_{mode}.json``.

Already-cached sentences are skipped, so the script is safe to re-run.

Usage:
    python tools/generate_llm_draft_gloss.py --modes train test
    python tools/generate_llm_draft_gloss.py --modes test --limit 10
    python tools/generate_llm_draft_gloss.py --model Qwen/Qwen2.5-72B-Instruct-AWQ
"""

import argparse
import json
import os
import sys
from tqdm import tqdm

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_DIR = os.path.join(PROJECT_ROOT, 'cache')
PROMPT_PATH = os.path.join(PROJECT_ROOT, 'prompts', 'pseudogloss_extraction_prompt.txt')


def load_prompt_template():
    with open(PROMPT_PATH, 'r', encoding='utf-8') as f:
        return f.read()


def load_sentences(mode, root_dir):
    import pandas as pd
    xlsx = os.path.join(root_dir, f'how2sign_realigned_{mode}.xlsx')
    if not os.path.exists(xlsx):
        raise FileNotFoundError(f"Metadata not found: {xlsx}")
    df = pd.read_excel(xlsx)
    return df['SENTENCE'].dropna().tolist()


def build_model(model_name, device_map='auto'):
    print(f"Loading model: {model_name} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map=device_map,
        trust_remote_code=True,
    )
    model.eval()
    print(f"Model loaded. Device map: {model.hf_device_map if hasattr(model, 'hf_device_map') else 'single'}")
    return model, tokenizer


def extract_gloss(model, tokenizer, prompt_template, sentence):
    prompt = prompt_template.replace('{input_sentence}', sentence)

    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    inputs = tokenizer(text, return_tensors='pt').to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=False,
            temperature=None,
            top_p=None,
        )

    # Decode only the generated part
    new_tokens = output_ids[0, inputs['input_ids'].shape[1]:]
    raw = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    # Take first line, lowercase, clean
    gloss = raw.split('\n')[0].strip().lower()
    return gloss


def main():
    parser = argparse.ArgumentParser(description="Generate LLM draft pseudo-glosses (local)")
    parser.add_argument("--modes", nargs='+', default=['train', 'test'],
                        choices=['train', 'val', 'test'])
    parser.add_argument("--model", type=str, default='Qwen/Qwen2.5-32B-Instruct',
                        help="HuggingFace model name")
    parser.add_argument("--root_dir", type=str,
                        default='/scratch/rhong5/dataset/Neural-Sign-Actors')
    parser.add_argument("--limit", type=int, default=None,
                        help="Process only first N sentences (for testing)")
    parser.add_argument("--force", action="store_true",
                        help="Re-generate even if cached")
    parser.add_argument("--save_every", type=int, default=100,
                        help="Save cache every N sentences")
    args = parser.parse_args()

    model, tokenizer = build_model(args.model)

    os.makedirs(CACHE_DIR, exist_ok=True)
    prompt_template = load_prompt_template()

    for mode in args.modes:
        cache_path = os.path.join(CACHE_DIR, f'llm_draft_gloss_{mode}.json')

        cache = {}
        if os.path.exists(cache_path) and not args.force:
            with open(cache_path, 'r', encoding='utf-8') as f:
                cache = json.load(f)
            print(f"[{mode}] Loaded {len(cache)} cached entries")

        sentences = load_sentences(mode, args.root_dir)
        if args.limit:
            sentences = sentences[:args.limit]

        unique_sentences = list(set(sentences))
        to_process = [s for s in unique_sentences if s not in cache]
        print(f"[{mode}] {len(unique_sentences)} unique, {len(to_process)} to process")

        if not to_process:
            print(f"[{mode}] All cached, skipping.")
            continue

        for i, sentence in enumerate(tqdm(to_process, desc=f"[{mode}]")):
            gloss = extract_gloss(model, tokenizer, prompt_template, sentence)
            cache[sentence] = gloss

            if (i + 1) % args.save_every == 0 or i == len(to_process) - 1:
                tmp_path = cache_path + '.tmp'
                with open(tmp_path, 'w', encoding='utf-8') as f:
                    json.dump(cache, f, ensure_ascii=False, indent=2)
                os.replace(tmp_path, cache_path)

            if (i + 1) % 200 == 0:
                print(f"  '{sentence[:50]}' → '{gloss}'")

        print(f"[{mode}] Done. {len(cache)} total entries in {cache_path}")

    print("\n=== Sample outputs ===")
    for sent, gloss in list(cache.items())[:5]:
        print(f"  SENT:  {sent[:100]}")
        print(f"  GLOSS: {gloss}")
        print()


if __name__ == '__main__':
    main()
