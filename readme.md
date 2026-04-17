# How2Sign Vocabulary Matching Against ASL Gloss Datasets

## How2Sign Train Set Scale

| Metric | Count |
|---|---|
| Sentences | 31,165 |
| Total word tokens | 569,462 |
| Unique words (raw) | 14,880 |
| Unique lemmas (after preprocessing) | ~7,900–7,911 |

## Preprocessing Pipeline for H2S Unique Lemmas

Starting from 14,880 raw unique words, the vocabulary was reduced to ~7,911 lemmas through the following steps:

1. **Proper noun removal** — Mid-sentence capitalized words (place names, person names, brand names) were dropped using a capitalization heuristic. Only the first word of each sentence was exempt.

2. **Contraction expansion** — Contractions were split before tokenization: `don't → do not`, `we're → we are`, `I've → I have`, `he'll → he will`, `I'd → I would`, `I'm → I am`, `word's → word` (possessive dropped).

3. **Lemmatization** — Inflected forms were reduced to their base form:
   - Irregular verbs: `went → go`, `made → make`, `ran → run`, etc. (explicit lookup table)
   - Regular verb inflections: `-ing`, `-ed`, `-es`, `-s` suffixes stripped
   - Plural nouns: `-s`, `-es`, `-ies` → singular
   - Comparatives / superlatives: `-er`, `-est`, `-ier`, `-iest` → base adjective
   - Adverbs: `-ly` → base adjective (`slowly → slow`, `nicely → nice`)

4. **English dictionary filter** — Lemmas not found in the American English system dictionary (73,604 entries) were discarded, removing noise, abbreviations, and tokenization artifacts.

## Matching Results

Each H2S lemma was matched against the gloss/keyword vocabulary of three ASL datasets using two methods:
- **Direct match**: exact string match between H2S lemma and dataset keyword
- **Synonym match**: if no direct match, up to 5 WordNet synonyms of the H2S lemma were checked against the keyword set

| Dataset | Keywords | H2S Lemmas | Direct Match | Synonym Match | **Total Matched** | No Match |
|---|---|---|---|---|---|---|
| WLASL | 2,000 | 7,376 | 1,693 (23.0%) | 804 (10.9%) | **2,497 (33.9%)** | 4,879 (66.1%) |
| ASL-LEX | 2,017 | 7,391 | 1,674 (22.6%) | 843 (11.4%) | **2,517 (34.1%)** | 4,874 (65.9%) |
| ASL-SignBank | 7,356 | 7,911 | 4,069 (51.4%) | 616 (7.8%) | **4,685 (59.2%)** | 3,226 (40.8%) |

## Key Observations

- **ASL-SignBank is the best choice** as a gloss lookup library, covering 59.2% of H2S lemmas — nearly double WLASL and ASL-LEX. This is because SignBank's `Keywords` field contains rich synonym lists per gloss entry, giving much broader vocabulary coverage.
- **WLASL and ASL-LEX perform similarly** (~34%), as both contain ~2,000 gloss entries with limited synonym coverage.
- The remaining unmatched lemmas in ASL-SignBank fall into two categories: (1) function words (`of`, `an`, `would`) which are not expected to have gloss entries, and (2) domain-specific How2Sign vocabulary (`wax`, `clay`, `fabric`, `garlic`) covering crafts and cooking topics absent from the ASL gloss dictionaries.