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
        self._gloss_strings = self._load_llm_draft_cache(mode, cfg)

    def _load_llm_draft_cache(self, mode, cfg=None):
        gloss_source = getattr(cfg, 'GLOSS_SOURCE', 'llm_draft') if cfg is not None else 'llm_draft'
        if gloss_source == 'llm_shuffled':
            cache_name = f'llm_draft_gloss_shuffled_{mode}.json'
        else:
            cache_name = f'llm_draft_gloss_{mode}.json'
        cache_path = os.path.join(CACHE_DIR, cache_name)
        if not os.path.exists(cache_path):
            raise FileNotFoundError(
                f"LLM draft cache not found at {cache_path}."
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
        result = super().__getitem__(idx)
        if self.use_fk_cache:
            seq, sentence, _, length, gt_joints44 = result
            gloss_string = self._gloss_strings[idx] if length > 0 else ''
            return seq, sentence, gloss_string, length, gt_joints44
        seq, sentence, _, length = result
        gloss_string = self._gloss_strings[idx] if length > 0 else ''
        return seq, sentence, gloss_string, length
