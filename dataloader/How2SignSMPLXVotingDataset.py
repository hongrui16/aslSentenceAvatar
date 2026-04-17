"""
How2SignSMPLXVotingDataset
==========================
Extends How2SignSMPLXDataset with LLM-generated draft pseudo-gloss strings
loaded from a pre-computed cache (``./cache/llm_draft_gloss_{mode}.json``).

Generate the cache first:
    python tools/generate_llm_draft_gloss.py --modes train val test

__getitem__ returns (seq, sentence, llm_draft_gloss, length).
"""
import json
import os
import sys

if __name__ == "__main__" and __package__ is None:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataloader.How2SignSMPLXDataset import How2SignSMPLXDataset

CACHE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'cache'
)


class How2SignSMPLXVotingDataset(How2SignSMPLXDataset):

    def __init__(self, mode='train', cfg=None, logger=None):
        super().__init__(mode=mode, cfg=cfg, logger=logger)
        self._gloss_strings = self._load_llm_draft_cache(mode)

    def _load_llm_draft_cache(self, mode):
        cache_path = os.path.join(CACHE_DIR, f'llm_draft_gloss_{mode}.json')
        if not os.path.exists(cache_path):
            raise FileNotFoundError(
                f"LLM draft cache not found at {cache_path}. "
                f"Run: python tools/generate_llm_draft_gloss.py --modes {mode}"
            )
        with open(cache_path, 'r', encoding='utf-8') as f:
            cache = json.load(f)

        gloss_strings = []
        n_miss = 0
        for sentence, _ in self.data_list:
            g = cache.get(sentence)
            if g is None:
                g = ''
                n_miss += 1
            gloss_strings.append(g)

        if self.logger is not None:
            example = next((g for g in gloss_strings if g), '')
            self.logger.info(
                f"[{mode}] loaded {len(gloss_strings)} LLM draft glosses from {cache_path} "
                f"({n_miss} missing; example: '{example}')"
            )
        return gloss_strings

    def __getitem__(self, idx):
        seq, sentence, _, length = super().__getitem__(idx)
        gloss_string = self._gloss_strings[idx] if length > 0 else ''
        return seq, sentence, gloss_string, length
