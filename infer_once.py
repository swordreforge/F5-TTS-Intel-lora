#!/usr/bin/env python3
"""Single-shot F5-TTS inference worker (argv-isolated, crash-safe).
Called by infer_32uwgy.py via subprocess; args pass through a JSON file
to avoid all shell-quoting issues with CJK text.
"""

import json
import sys

import soundfile as sf
import torch
import torchaudio


def _load(path, *args, **kwargs):
    data, sr = sf.read(path, dtype="float32", always_2d=True)
    return torch.from_numpy(data.T), sr


def _info(path, *args, **kwargs):
    info = sf.info(path)
    return type("I", (), {"sample_rate": info.samplerate, "num_frames": info.frames})()


torchaudio.load = _load
torchaudio.info = _info

params = json.loads(open(sys.argv[1], encoding="utf-8").read())

argv = [
    "infer_cli",
    "--model",
    "F5TTS_v1_Base",
    "--ref_audio",
    params["ref_audio"],
    "--ref_text",
    params["ref_text"],
    "--gen_text",
    params["gen_text"],
    "--output_dir",
    params["out_dir"],
    "--output_file",
    params["out_file"],
    "--device",
    "xpu",
    "--speed",
    str(params["speed"]),
    "--nfe_step",
    str(params.get("nfe", 16)),
]
if params.get("ckpt"):
    argv += ["--ckpt_file", params["ckpt"]]
sys.argv = argv

from f5_tts.infer.infer_cli import main  # noqa: E402 -- argv must be set before this import-time parse


main()
