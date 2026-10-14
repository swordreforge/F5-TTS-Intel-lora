"""XPU smoke training wrapper: soundfile fallback for torchaudio (torchcodec is
CUDA-built and cannot load on torch 2.15 XPU), then run finetune_cli.main."""

import sys

import soundfile as sf
import torch
import torchaudio
from f5_tts.train.finetune_cli import main as _finetune_main


def _load(path, *args, **kwargs):
    data, sr = sf.read(path, dtype="float32", always_2d=True)
    return torch.from_numpy(data.T), sr


def _info(path, *args, **kwargs):
    info = sf.info(path)
    return type("I", (), {"sample_rate": info.samplerate, "num_frames": info.frames})()


def main():
    torchaudio.load = _load
    torchaudio.info = _info

    sys.argv = [
        "finetune_cli",
        "--exp_name",
        "F5TTS_v1_Base",
        "--finetune",
        "--dataset_name",
        "XPU_SMOKE",
        "--epochs",
        "1",
        "--batch_size_per_gpu",
        "3200",
        "--max_samples",
        "2",
        "--grad_accumulation_steps",
        "1",
        "--num_warmup_updates",
        "5",
        "--save_per_updates",
        "2",
        "--last_per_updates",
        "2",
        "--keep_last_n_checkpoints",
        "1",
        "--learning_rate",
        "1e-5",
    ]
    _finetune_main()


if __name__ == "__main__":
    main()
