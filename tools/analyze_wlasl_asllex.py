import json
import csv

# ============================================================
# Part 1: WLASL Per-Gloss Video Distribution
# ============================================================
with open('/home/rhong5/research_pro/hand_modeling_pro/aslAvatar/data/WLASL/WLASL_v0.3.json') as f:
    wlasl = json.load(f)

# {gloss: num_videos}
gloss_counts = {entry['gloss']: len(entry['instances']) for entry in wlasl}

# 按图片中的阈值生成分布表
thresholds = list(range(1, 61, 5))  # [1, 6, 11, 16, 21, 26, 31, 36, 41, 46, 51, 56]

print("=" * 65)
print("WLASL Per-Gloss Video Distribution")
print(f"{'Min Videos per Gloss':>22}  {'Qualifying Glosses':>20}  {'Total Videos':>14}")
print("-" * 65)
for t in thresholds:
    qualifying = {g: c for g, c in gloss_counts.items() if c >= t}
    print(f"{'≥ ' + str(t):>22}  {len(qualifying):>20,}  {sum(qualifying.values()):>14,}")
print("=" * 65)

# ============================================================
# Part 2: ASL-LEX ∩ WLASL cross-reference
# ============================================================
asl_lex_glosses = []
with open('/home/rhong5/research_pro/hand_modeling_pro/aslAvatar/data/ASL_LEX2.0/ASL-LEX View Data.csv', 'r', encoding='utf-8') as f:
    reader = csv.reader(f)
    header = next(reader)  # skip header row
    for row in reader:
        if row:
            asl_lex_glosses.append(row[0].strip())

# case-insensitive matching
wlasl_lower = {g.lower(): g for g in gloss_counts}
asl_lex_set = set(g.lower() for g in asl_lex_glosses)

overlap = asl_lex_set & set(wlasl_lower.keys())

# 重叠gloss的视频数量
overlap_counts = {wlasl_lower[g]: gloss_counts[wlasl_lower[g]] for g in overlap}

print(f"\nASL-LEX 总 gloss 数: {len(asl_lex_set)}")
print(f"WLASL 总 gloss 数:   {len(gloss_counts)}")
print(f"重叠 gloss 数:       {len(overlap)}")
print(f"ASL-LEX 覆盖率:      {len(overlap)/len(asl_lex_set)*100:.1f}%")

print(f"\n{'=' * 65}")
print("ASL-LEX ∩ WLASL Per-Gloss Video Distribution")
print(f"{'Min Videos per Gloss':>22}  {'Qualifying Glosses':>20}  {'Total Videos':>14}")
print("-" * 65)
for t in thresholds:
    qualifying = {g: c for g, c in overlap_counts.items() if c >= t}
    print(f"{'≥ ' + str(t):>22}  {len(qualifying):>20,}  {sum(qualifying.values()):>14,}")
print("=" * 65)