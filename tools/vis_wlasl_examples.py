import json
import os
import base64

# The 10 glosses with lowest retention from ASL3DWord
lowest_kept = [
    ("computer", 8, 30, "27%"),
    ("later",    6, 20, "30%"),
    ("cow",      6, 19, "32%"),
    ("cool",     7, 21, "33%"),
    ("same",     6, 18, "33%"),
    ("right",    7, 18, "39%"),
    ("shirt",    8, 20, "40%"),
    ("cheat",    8, 18, "44%"),
    ("full",     8, 18, "44%"),
    ("clothes",  11, 25, "44%"),
]

# Load WLASL
with open('/scratch/rhong5/dataset/wlasl/WLASL_v0.3.json', 'r') as f:
    wlasl = json.load(f)

gloss_map = {entry['gloss']: entry['instances'] for entry in wlasl}

html = """<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>WLASL Lowest Retention Glosses</title>
<style>
  body { font-family: Arial, sans-serif; background: #1a1a1a; color: #fff; padding: 20px; }
  h2 { margin-top: 30px; }
  h2 span.stats { font-size: 14px; color: #aaa; font-weight: normal; }
  hr { border: 1px solid #444; margin: 30px 0; }
  .grid { display: flex; flex-wrap: wrap; gap: 10px; }
  .cell { width: 18%; text-align: center; }
  .cell video { width: 100%; border-radius: 4px; }
  .cell p { font-size: 12px; color: #aaa; margin: 4px 0; }
</style>
</head><body>
<h1>WLASL: 10 Glosses with Lowest Retention in ASL3DWord</h1>
"""

for gloss, kept, total, pct in lowest_kept:
    instances = gloss_map.get(gloss, [])
    html += f'<hr>\n<h2>{gloss} <span class="stats">ASL3DWord: {kept} / WLASL: {total} (Kept {pct})</span></h2>\n'
    html += '<div class="grid">\n'
    for inst in instances:
        vid = inst['video_id']
        # find video file
        path = None
        for ext in ['mp4', 'swf', 'mov']:
            p = f"/scratch/rhong5/dataset/wlasl/videos/{vid}.{ext}"
            if os.path.exists(p):
                path = p
                break
        if path is None:
            # html += f'  <div class="cell"><p>id: {vid} (missing)</p></div>\n'
            continue
        # read and encode as base64
        with open(path, 'rb') as vf:
            b64 = base64.b64encode(vf.read()).decode('utf-8')
        mime = 'video/mp4' if path.endswith('.mp4') else 'video/quicktime'
        html += f'  <div class="cell">\n'
        html += f'    <video controls muted preload="metadata" onloadedmetadata="this.playbackRate=0.5" src="data:{mime};base64,{b64}"></video>\n'        
        html += f'    <p>id: {vid}</p>\n'
        html += f'  </div>\n'
    html += '</div>\n'

html += '</body></html>'

with open('lowest_retention.html', 'w') as f:
    f.write(html)

print("Done! File size:", os.path.getsize('lowest_retention.html') / 1024 / 1024, "MB")