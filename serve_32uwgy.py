#!/usr/bin/env python3
"""32uwgy TTS API (resident model, Intel XPU).
Endpoints:
  GET  /health            -> {"status": "ok", "model": ...}
  POST /tts               -> wav audio
    {"text": "...", "speed": 0.8, "nfe": 16, "verify": false, "seed_ref": false}
    verify=true enables the ASR repetition gate (slower, best-of-2).
Run: f5tts/bin/python serve_32uwgy.py [--port 8899]
Test: curl -X POST localhost:8899/tts -H 'Content-Type: application/json' \
        -d '{"text":"海底的洋流正在变强。"}' --output out.wav
"""

import argparse
import asyncio
import io
import re
import uuid
from pathlib import Path

import soundfile as sf
import torch
import torchaudio
from fastapi import FastAPI, Response
from fastapi.responses import StreamingResponse
from pydantic import BaseModel


def _load(path, *args, **kwargs):
    data, sr = sf.read(path, dtype="float32", always_2d=True)
    return torch.from_numpy(data.T), sr


def _info(path, *args, **kwargs):
    info = sf.info(path)
    return type("I", (), {"sample_rate": info.samplerate, "num_frames": info.frames})()


def install_shims():
    torchaudio.load = _load
    torchaudio.info = _info


install_shims()


BASE = Path("/home/swordreforge/F5-TTS")
REF_AUDIO = str(BASE / "data/teacher_wavs/extend3.clean.miss_0047.wav")
REF_TEXT = "截至二零二三年六月，全球米柚月活跃用户六点零六亿。"
FINETUNED_CKPT = str(BASE / "ckpts/TEACHER32/model_last.pt")
VOCAB = str(BASE / "src/f5_tts/infer/examples/vocab.txt")

app = FastAPI(title="32uwgy TTS")
api = None
asr_model = None
# resident preprocessed ref (avoids per-request reload)
ref_audio_t = None
ref_sr = 24000
ref_text_used = REF_TEXT


class TTSReq(BaseModel):
    text: str
    speed: float = 0.8
    nfe: int = 8
    cfg: float = 2.0
    verify: bool = False


def nopunct(s: str) -> str:
    return re.sub(r"[，。！？；：、…—\"'「」『』（）()·\s]", "", s)


def repeat_len(transcript: str, ref: str) -> int:
    t, r = nopunct(transcript), nopunct(ref)
    for k in range(min(20, len(r), len(t)), 0, -1):
        if t.startswith(r[-k:]):
            return k
    return 0


def get_asr():
    global asr_model
    if asr_model is None:
        import av

        _o = av.open
        av.open = lambda *a, **k: _o(*a, **{kk: vv for kk, vv in k.items() if kk != "metadata_errors"})
        from faster_whisper import WhisperModel

        asr_model = WhisperModel("small", device="cpu", compute_type="int8")
    return asr_model


def synth_sync(text: str, speed: float, nfe: int, cfg: float) -> tuple[bytes, int]:
    wav, sr, _ = api.infer(
        ref_file=REF_AUDIO,
        ref_text=REF_TEXT,
        gen_text=text,
        speed=speed,
        nfe_step=nfe,
        cfg_strength=cfg,
        show_info=lambda *a, **k: None,
        progress=None,
    )
    torch.xpu.synchronize()
    buf = io.BytesIO()
    sf.write(buf, wav, sr, format="WAV")
    return buf.getvalue(), sr


def transcribe_sync(wav_bytes: bytes) -> str:
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(wav_bytes)
        path = f.name
    try:
        segs, _ = get_asr().transcribe(path, language="zh", beam_size=5)
        return "".join(s.text for s in segs)
    finally:
        Path(path).unlink(missing_ok=True)


@app.on_event("startup")
def startup():
    global api, ref_audio_t, ref_sr
    from f5_tts.api import F5TTS
    from f5_tts.infer.utils_infer import preprocess_ref_audio_text

    api = F5TTS(model="F5TTS_v1_Base", ckpt_file=FINETUNED_CKPT, vocab_file=VOCAB, device="xpu")
    ref_file, _ = preprocess_ref_audio_text(REF_AUDIO, REF_TEXT, show_info=lambda *a, **k: None)
    ref_audio_t, ref_sr = torchaudio.load(ref_file)
    # warmup: pay oneDNN JIT + caches upfront (GSV lesson: 启动预热)
    api.infer(ref_file=REF_AUDIO, ref_text=REF_TEXT, gen_text="预热。", show_info=lambda *a, **k: None, progress=None)
    torch.xpu.synchronize()
    print("MODEL READY", flush=True)


@app.get("/health")
def health():
    return {"status": "ok" if api else "loading", "model": "TEACHER32/finetuned"}


@app.post("/tts")
async def tts(req: TTSReq):
    if not req.text.strip():
        return Response(content=b"empty text", status_code=400)
    tag = uuid.uuid4().hex[:8]
    for attempt in range(2 if req.verify else 1):
        wav_bytes, sr = await asyncio.to_thread(synth_sync, req.text, req.speed, req.nfe, req.cfg)
        if not req.verify:
            break
        tr = await asyncio.to_thread(transcribe_sync, wav_bytes)
        k = repeat_len(tr, REF_TEXT)
        print(f"[{tag} try{attempt}] repeat_len={k} asr_head={tr[:30]}", flush=True)
        if k < 6:
            break
    return Response(content=wav_bytes, media_type="audio/wav")


@app.post("/tts/stream")
def tts_stream(req: TTSReq):
    """Chunked int16 PCM stream (24kHz mono). First text piece is kept tiny
    so the first audio chunk arrives fast (socket_server few/min recipe).
    curl: curl -X POST localhost:8899/tts/stream -H 'Content-Type: application/json'
            -d '{"text":"..."}' --output out.pcm
          ffmpeg -f s16le -ar 24000 -ac 1 -i out.pcm out.wav"""
    from f5_tts.infer.utils_infer import chunk_text, infer_batch_process

    def gen():
        batches = chunk_text(req.text, max_chars=200)
        # fast first package: sub-split head into tiny pieces
        if batches:
            head = chunk_text(batches[0], max_chars=50)
            head = chunk_text(head[0], max_chars=15) + head[1:] if head else head
            batches = head + batches[1:]
        stream = infer_batch_process(
            (ref_audio_t, ref_sr),
            ref_text_used,
            batches,
            api.ema_model,
            api.vocoder,
            api.mel_spec_type,
            progress=None,
            nfe_step=req.nfe,
            cfg_strength=req.cfg,
            speed=req.speed,
            device=api.device,
            streaming=True,
            chunk_size=2048,
        )
        import numpy as np

        for audio_chunk, _ in stream:
            if audio_chunk is not None and len(audio_chunk) > 0:
                yield (np.int16(np.clip(audio_chunk, -1, 1) * 32767)).tobytes()

    return StreamingResponse(gen(), media_type="audio/x-pcm")


if __name__ == "__main__":
    import uvicorn

    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8899)
    args = ap.parse_args()
    uvicorn.run(app, host="0.0.0.0", port=args.port, workers=1)
