#!/usr/bin/env python3
"""F5-TTS XPU steady-state benchmark: load once, synth twice, split timings."""

import sys
import time

import soundfile as sf
import torch
import torchaudio


def _load(path, *a, **k):
    data, sr = sf.read(path, dtype="float32", always_2d=True)
    return torch.from_numpy(data.T), sr


torchaudio.load = _load

t0 = time.time()
from f5_tts.api import F5TTS


api = F5TTS(
    model="F5TTS_v1_Base",
    ckpt_file="/home/swordreforge/F5-TTS/ckpts/TEACHER32/model_last.pt",
    vocab_file="/home/swordreforge/F5-TTS/src/f5_tts/infer/examples/vocab.txt",
    device="xpu",
)
t_load = time.time() - t0
print(f"MODEL LOAD: {t_load:.1f}s", flush=True)

REF = "/home/swordreforge/F5-TTS/data/teacher_wavs/extend3.clean.miss_0047.wav"
RT = "截至二零二三年六月，全球米柚月活跃用户六点零六亿。"
for i, text in enumerate(
    [
        "海底的洋流正在变强，我们必须马上重构防护结构。",
        "社区不仅提供了丰富的教程和学习资源，还鼓励会员们分享创意。",
    ]
):
    t1 = time.time()
    wav, sr, spec = api.infer(
        ref_file=REF,
        ref_text=RT,
        gen_text=text,
        show_info=print,
        progress=None,
    )
    torch.xpu.synchronize()
    dt = time.time() - t1
    dur = len(wav) / sr
    sf.write(f"/tmp/bench{i}.wav", wav, sr)
    print(f"SYNTH{i}: wall={dt:.1f}s audio={dur:.1f}s RTF={dt / dur:.2f}", flush=True)
