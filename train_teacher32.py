#!/usr/bin/env python3
"""Formal finetune on TEACHER32 (QORA-TTS 32uwgy teacher data) via XPU.
- torchaudio shims -> soundfile (torchcodec is CUDA-built, unloadable on XPU).
- force DataLoader num_workers=2 (default 16 persistent workers OOMs 30G box).
- batch 1600 frames x grad_accum 2 (= same effective 3200 as smoke-tested).
~110 updates/epoch on 0.53h data; 25 epochs ~= 2800 updates.
"""
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

from f5_tts.model import trainer as _trainer_mod

_orig_train = _trainer_mod.Trainer.train


def _lean_train(self, dataset, num_workers=1, **kwargs):
    return _orig_train(self, dataset, num_workers=num_workers, **kwargs)


_trainer_mod.Trainer.train = _lean_train

sys.argv = [
    "finetune_cli",
    "--exp_name", "F5TTS_v1_Base",
    "--finetune",
    "--dataset_name", "TEACHER32",
    "--epochs", "25",
    "--batch_size_per_gpu", "1600",
    "--max_samples", "32",
    "--grad_accumulation_steps", "2",
    "--num_warmup_updates", "200",
    "--save_per_updates", "200",
    "--last_per_updates", "100",
    "--keep_last_n_checkpoints", "3",
    "--learning_rate", "1e-5",
]

from f5_tts.train.finetune_cli import main

main()
