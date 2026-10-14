#!/usr/bin/env python3
"""Clean raw article txt -> TTS-friendly one-sentence-per-line txt.
Drops: markdown, URLs, date/author lines, code fragments, too-short bits.
Splits long sentences at ，；：、 so every line <= MAX_CHARS (=> <30s audio).
Usage: clean_texts.py <in1.txt> [in2.txt ...] <out_dir>
"""
import re
import sys
from pathlib import Path

MAX_CHARS = 100
MIN_CHARS = 8

DROP_PATTERNS = [
    r'https?://',            # URLs / 原文链接
    r'^\s*原文链接',          # source-link lines
    r'发表于\s*\d{4}',       # author/date lines
    r'^\s*#{1,6}\s',         # markdown headers
    r'^\s*>\s?',             # quotes
    r'^\s*[-*+]\s+',         # list items
    r'^\s*\|',               # tables
    r'^\s*```',              # code fences
    r'^\s*\d+\.\s+[A-Za-z]', # ascii list
]

CJK = '[一-鿿]'


def clean_line(line: str) -> str:
    line = line.strip()
    for p in DROP_PATTERNS:
        if re.search(p, line):
            return ''
    # strip markdown decorations, keep text
    line = re.sub(r'\*\*(.+?)\*\*', r'\1', line)
    line = re.sub(r'`(.+?)`', r'\1', line)
    line = line.replace('【', '').replace('】', '')
    # drop long all-english parenthetical expansions, keep short ones like (TTS)
    def _paren(m):
        inner = m.group(1)
        return m.group(0) if len(inner) <= 12 else ''
    line = re.sub(r'\(([A-Za-z][A-Za-z0-9\s.,\-&]*)\)', _paren, line)
    line = re.sub(r'\s+', ' ', line).strip()
    line = re.sub(r'(?<=[一-鿿]) | (?=[一-鿿])', '', line)
    # drop lines without CJK (pure english/code/metadata)
    if not re.search(CJK, line):
        return ''
    return line


def split_long(sent: str):
    if len(sent) <= MAX_CHARS:
        return [sent]
    parts, buf = [], ''
    for ch in sent:
        buf += ch
        if ch in '，；：、' and len(buf) >= 40:
            parts.append(buf)
            buf = ''
    if buf:
        parts.append(buf)
    # every part must be standalone (end with 。) so downstream never rejoins
    fixed = []
    for p in parts:
        if p[-1] not in '。！？…':
            p += '。'
        fixed.append(p)
    parts = fixed
    # hard cap: split any remainder over MAX_CHARS at last break opportunity
    capped = []
    for p in parts:
        while len(p) > MAX_CHARS:
            cut = max([p.rfind(m, 0, MAX_CHARS) for m in '，；：、 '] + [0])
            if cut <= 20:
                cut = MAX_CHARS
                tail = p[cut:]
            else:
                tail = p[cut + 1:] if p[cut] in '，；：、' else p[cut:]
                p = p[:cut + 1] if p[cut] in '，；：、' else p[:cut]
                capped.append(p)
                p = tail.lstrip('，；：、 ')
                continue
            capped.append(p[:cut])
            p = tail.lstrip('，；：、 ')
        if p:
            capped.append(p)
    parts = capped
    # merge trailing stub
    if len(parts) > 1 and len(parts[-1]) < MIN_CHARS:
        parts[-2] += parts[-1]
        parts.pop()
    return [p for p in parts if len(p) >= MIN_CHARS]


def main():
    *inputs, out_dir = sys.argv[1:]
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    total = 0
    for inp in inputs:
        sents = []
        for raw in Path(inp).read_text(encoding='utf-8').splitlines():
            line = clean_line(raw)
            if not line:
                continue
            for s in re.split(r'(?<=[。！？…])', line):
                s = s.strip()
                if not s:
                    continue
                if not s[-1] in '。！？…':
                    s += '。'
                if len(s) < MIN_CHARS:
                    continue
                sents.extend(split_long(s))
        out = out_dir / (Path(inp).stem + '.clean.txt')
        out.write_text('\n'.join(sents) + '\n', encoding='utf-8')
        chars = sum(len(s) for s in sents)
        print(f'{inp}: {len(sents)} sents, {chars} chars -> {out.name} (~{chars/4/60:.1f} min audio)')
        total += chars
    print(f'TOTAL ~{total/4/60:.1f} min audio')


if __name__ == '__main__':
    main()
