#!/usr/bin/env python3
"""Recover teacher dataset: drop truncated/rejoined clips, emit missing-line files."""
import csv
import re
from pathlib import Path

BASE = Path('/home/swordreforge/F5-TTS')
WAV = BASE / 'data/teacher_wavs'
CSV = BASE / 'teacher_metadata.csv'
CLEAN = BASE / 'data/cleaned'

QR = BASE / '下载/QORA-TTS-12Hz-1.7B/result.txt'  # not used; result path below
RESULT = Path('/home/swordreforge/下载/QORA-TTS-12Hz-1.7B/result.txt')

rows = list(csv.DictReader(CSV.open(encoding='utf-8'), delimiter='|'))
print('csv rows:', len(rows))

# new-scheme lines per source
new_lines = {}
for f in sorted(CLEAN.glob('*.clean.txt')):
    new_lines[f.stem] = [l for l in f.read_text(encoding='utf-8').splitlines() if l.strip()]
newset = {l for v in new_lines.values() for l in v}

# result.txt split (old scheme source)
rt = RESULT.read_text(encoding='utf-8')
rt_sents = [s.strip() for s in re.split(r'(?<=[。！？…])', rt) if s.strip()]

drop_idx, drop_reason = set(), {}

# 1. truncated: known bad clips
for i, r in enumerate(rows):
    name = Path(r['audio_file']).name
    if name in ('clip_0022.wav', 'clip_0079.wav'):
        drop_idx.add(i)
        drop_reason[name] = 'truncated@40s'

# 2. rejoined: old text not present as a standalone new line (and not from result.txt)
rtset = set(rt_sents)
for i, r in enumerate(rows):
    if i in drop_idx:
        continue
    name = Path(r['audio_file']).name
    if r['text'] not in newset and r['text'] not in rtset:
        drop_idx.add(i)
        drop_reason[name] = 'rejoined'

print('dropping:', len(drop_idx))
for k, v in sorted(drop_reason.items()):
    print(' ', k, v)

keep = [r for i, r in enumerate(rows) if i not in drop_idx]
covered = {r['text'] for r in keep}

# delete wav + log of dropped
for i in drop_idx:
    for suf in ('', '.log'):
        p = Path(rows[i]['audio_file'] + suf)
        if p.exists():
            p.unlink()

# rewrite csv
with CSV.open('w', encoding='utf-8', newline='') as f:
    w = csv.DictWriter(f, fieldnames=['audio_file', 'text'], delimiter='|')
    w.writeheader()
    w.writerows(keep)

# missing: new lines not covered -> per-source files
MISS = BASE / 'data/missing'
MISS.mkdir(exist_ok=True)
total = 0
for stem, lines in new_lines.items():
    miss = [l for l in lines if l not in covered]
    if miss:
        (MISS / f'{stem}.miss.txt').write_text('\n'.join(miss) + '\n', encoding='utf-8')
    print(f'{stem}: {len(lines)} lines, missing {len(miss)}')
    total += len(miss)

# result.txt truncated clip_0022 halves
clip22 = next(r['text'] for r in rows if Path(r['audio_file']).name == 'clip_0022.wav') \
    if any(Path(r['audio_file']).name == 'clip_0022.wav' for r in keep) else None
# (already dropped above) -> find original text from rt_sents by index 21 (0-based 21)
orig22 = rt_sents[21]
print('clip22 orig len:', len(orig22))
halves, buf = [], ''
for ch in orig22:
    buf += ch
    if ch in '，；：、' and len(buf) >= 40:
        halves.append(buf)
        buf = ''
if buf:
    halves.append(buf)
halves = [h if h[-1] in '。！？…' else h + '。' for h in halves]
(MISS / 'result22.miss.txt').write_text('\n'.join(halves) + '\n', encoding='utf-8')
print('result22 halves:', len(halves))
total += len(halves)
print('TOTAL missing:', total)
