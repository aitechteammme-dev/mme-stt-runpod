import base64
import io
import os
import time
import traceback

import numpy as np
import runpod
import soundfile as sf
import torch
from scipy.signal import resample_poly
from transformers import pipeline

MODEL_ID = os.getenv("MODEL_ID", "mmeog/mme_stt_model")
MODEL_REVISION = os.getenv("MODEL_REVISION", "")

# Set to 0 to disable chunking. Utterances on a phone call are short, so keep it 0.
CHUNK_LENGTH_S = int(os.getenv("CHUNK_LENGTH_S", "0"))

# Language token for Whisper. Empty = auto-detect (slower, can mis-detect).
# TEST both "hi" and "en" on real clips: Hinglish (Roman script) models often
# work best with "en"; Devanagari output works best with "hi".
WHISPER_LANGUAGE = os.getenv("WHISPER_LANGUAGE", "").strip()

TARGET_SR = 16000
USE_CUDA = torch.cuda.is_available()


def log(msg):
    print(msg, flush=True)


# --------------------------------------------------------------------------
# Load the model ONCE at import time (not inside the request)
# --------------------------------------------------------------------------
def build_pipeline():
    kwargs = {
        "model": MODEL_ID,
        "device": 0 if USE_CUDA else -1,
        "dtype": torch.float16 if USE_CUDA else torch.float32,
    }
    if MODEL_REVISION:
        kwargs["revision"] = MODEL_REVISION
    if CHUNK_LENGTH_S > 0:
        kwargs["chunk_length_s"] = CHUNK_LENGTH_S
    return pipeline("automatic-speech-recognition", **kwargs)


def make_generate_kwargs():
    gk = {"task": "transcribe", "num_beams": 1, "do_sample": False}
    if WHISPER_LANGUAGE:
        gk["language"] = WHISPER_LANGUAGE
    return gk


GENERATE_KWARGS = make_generate_kwargs()

_t = time.perf_counter()
ASR = build_pipeline()
log(
    f"[STT] model={MODEL_ID} loaded in {time.perf_counter() - _t:.2f}s "
    f"cuda={USE_CUDA} language={WHISPER_LANGUAGE or 'auto'}"
)


def transcribe(audio: np.ndarray) -> str:
    with torch.inference_mode():
        result = ASR(
            {"raw": audio, "sampling_rate": TARGET_SR},
            generate_kwargs=GENERATE_KWARGS,
        )
    if USE_CUDA:
        torch.cuda.synchronize()
    return result["text"].strip()


# Warm up CUDA kernels so the first real request is fast
_t = time.perf_counter()
_warm = (np.random.randn(TARGET_SR * 3) * 0.005).astype(np.float32)
for _ in range(2):
    try:
        transcribe(_warm)
    except Exception as exc:  # warmup must never kill the worker
        log(f"[STT] warmup error (ignored): {exc}")
log(f"[STT] warmup done in {time.perf_counter() - _t:.2f}s")


def load_audio(data):
    """Returns float32 mono 16 kHz audio. Accepts WAV (audio_b64) or raw PCM16."""
    if data.get("audio_pcm16_b64"):
        raw = base64.b64decode(data["audio_pcm16_b64"], validate=True)
        sample_rate = int(data.get("sample_rate", TARGET_SR))
        audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    elif data.get("audio_b64"):
        raw = base64.b64decode(data["audio_b64"], validate=True)
        audio, sample_rate = sf.read(io.BytesIO(raw), dtype="float32")
    else:
        raise ValueError("Provide audio_b64 (WAV) or audio_pcm16_b64 (raw PCM16)")

    if audio.size == 0:
        raise ValueError("The audio is empty")
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    if sample_rate != TARGET_SR:
        audio = resample_poly(audio, TARGET_SR, sample_rate).astype(np.float32)
    return np.ascontiguousarray(audio, dtype=np.float32)


def handler(job):
    data = job.get("input") or {}

    # Cheap ping for pre-warming from your server
    if data.get("warmup"):
        return {"ok": True}

    task = data.get("task", "stt")

    if task == "health":
        return {
            "status": "ok",
            "model": MODEL_ID,
            "loaded": True,
            "cuda_available": USE_CUDA,
            "gpu": torch.cuda.get_device_name(0) if USE_CUDA else None,
            "language": WHISPER_LANGUAGE or "auto",
        }

    if task != "stt":
        raise ValueError("Supported tasks: health, stt")

    total_start = time.perf_counter()
    try:
        t0 = time.perf_counter()
        audio = load_audio(data)
        preprocess_s = time.perf_counter() - t0

        t0 = time.perf_counter()
        text = transcribe(audio)
        inference_s = time.perf_counter() - t0

        total_s = time.perf_counter() - total_start
        log(
            f"[STT_TIMING] audio_s={len(audio) / TARGET_SR:.2f} "
            f"preprocess_s={preprocess_s:.3f} inference_s={inference_s:.3f} "
            f"total_s={total_s:.3f} text_chars={len(text)}"
        )

        return {
            "text": text,
            "model": MODEL_ID,
            "timing": {
                "inference_s": round(inference_s, 3),
                "total_s": round(total_s, 3),
            },
        }

    except Exception as exc:
        log(f"STT inference error: {exc}")
        traceback.print_exc()
        raise


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
