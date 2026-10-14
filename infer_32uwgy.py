#!/usr/bin/env python3
"""32uwgy voice inference wrapper: fixed good ref + ASR gate + best-of-N retry.

- Fixed canonical ref (teacher-synthesized, exact-match, verified clean).
- Each try runs stock infer_cli; ASR (faster-whisper small int8) checks whether
  the output starts with the ref tail (the known repetition mode).
- Keeps the first clean try, else the one with the shortest repeat.
- torchaudio shims -> soundfile (torchcodec is CUDA-built, unloadable on XPU).

Usage:
  f5tts/bin/python infer_32uwgy.py --text "要合成的中文" [--out out.wav]
  f5tts/bin/python infer_32uwgy.py --text-file article.txt [--ckpt base] [--speed 0.8] [--tries 3]
"""
import argparse
import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import soundfile as sf

BASE = Path("/home/swordreforge/F5-TTS")
# Verified-clean canonical ref (teacher clip, exact transcript)
REF_AUDIO = str(BASE / "data/teacher_wavs/extend3.clean.miss_0047.wav")
REF_TEXT = "截至二零二三年六月，全球米柚月活跃用户六点零六亿。"
FINETUNED_CKPT = str(BASE / "ckpts/TEACHER32/model_last.pt")


def nopunct(s: str) -> str:
    return re.sub(r"[，。！？；：、…—\"'「」『』（）()·\s]", "", s)


def repeat_len(transcript: str, ref: str) -> int:
    """Longest ref-tail (>=1 char) matching transcript start; 0 if clean."""
    t, r = nopunct(transcript), nopunct(ref)
    best = 0
    for k in range(min(20, len(r), len(t)), 0, -1):
        if t.startswith(r[-k:]):
            return k
    return best


def asr_transcribe(wav: str) -> str:
    import av

    _o = av.open
    av.open = lambda *a, **k: _o(*a, **{kk: vv for kk, vv in k.items() if kk != "metadata_errors"})
    from faster_whisper import WhisperModel

    model = asr_transcribe._model
    if model is None:
        model = WhisperModel("small", device="cpu", compute_type="int8")
        asr_transcribe._model = model
    segs, _ = model.transcribe(wav, language="zh", beam_size=5)
    return "".join(s.text for s in segs)


asr_transcribe._model = None


def run_once(gen_text: str, ckpt: str | None, speed: float, out: str) -> bool:
    """Run one synthesis in an isolated subprocess. Returns True on success."""
    params = {
        "ref_audio": REF_AUDIO, "ref_text": REF_TEXT, "gen_text": gen_text,
        "out_dir": str(Path(out).parent), "out_file": Path(out).name,
        "speed": speed, "ckpt": ckpt,
    }
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as f:
        json.dump(params, f, ensure_ascii=False)
        param_file = f.name
    try:
        r = subprocess.run(
            [str(BASE / "f5tts/bin/python"), str(BASE / "infer_once.py"), param_file],
            capture_output=True, text=True, timeout=900,
        )
        if r.returncode != 0:
            print("infer subprocess failed:")
            print(r.stderr[-2000:] if r.stderr else "")
            return False
        return Path(out).exists()
    except subprocess.TimeoutExpired:
        print("infer subprocess timed out")
        return False
    finally:
        Path(param_file).unlink(missing_ok=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", default="", help="single sentence to synthesize")
    ap.add_argument("--text-file", default="", help="file: one sentence per line")
    ap.add_argument("--ckpt", default="finetuned",
                    help="'finetuned' (default), 'base', or path to .pt")
    ap.add_argument("--speed", type=float, default=0.8)
    ap.add_argument("--tries", type=int, default=3)
    ap.add_argument("--out", default=str(BASE / "tests/out_32uwgy.wav"))
    args = ap.parse_args()

    ckpt = None if args.ckpt == "base" else (FINETUNED_CKPT if args.ckpt == "finetuned" else args.ckpt)
    if args.text_file:
        sens = [l.strip() for l in Path(args.text_file).read_text(encoding="utf-8").splitlines() if l.strip()]
        gen_text = "\n".join(sens)
    else:
        gen_text = args.text
    if not gen_text:
        ap.error("need --text or --text-file")

    best, best_k = None, 10**9
    with tempfile.TemporaryDirectory() as td:
        keep = str(Path(td) / "best.wav")
        for i in range(args.tries):
            cand = str(Path(td) / f"try{i}.wav")
            print(f"[try {i+1}/{args.tries}] synthesizing...", flush=True)
            if not run_once(gen_text, ckpt, args.speed, cand):
                print(f"[try {i+1}] synthesis failed, skipping")
                continue
            tr = asr_transcribe(cand)
            k = repeat_len(tr, REF_TEXT)
            dur = round(sf.info(cand).duration, 2)
            print(f"[try {i+1}] repeat_len={k} dur={dur}s asr_head={tr[:30]}")
            if k < best_k:
                best, best_k = tr, k
                shutil.copy(cand, keep)
                if k < 6:
                    shutil.copy(cand, args.out)
                    print(f"ACCEPT try {i+1}: {args.out}")
                    return
        shutil.copy(keep, args.out)
        print(f"ALL TRIED, keeping shortest repeat ({best_k} chars): {args.out}\ntranscript: {best}")


if __name__ == "__main__":
    main()
