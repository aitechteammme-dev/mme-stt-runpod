
import base64
import io
import os
import threading
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

# Set to 0 to disable chunking. Benchmark before enabling.
CHUNK_LENGTH_S = int(os.getenv("CHUNK_LENGTH_S", "0"))

_asr = None
_model_lock = threading.Lock()


def load_model():
    global _asr

    if _asr is not None:
        return _asr

    with _model_lock:
        if _asr is not None:
            return _asr

        use_cuda = torch.cuda.is_available()

        kwargs = {
            "model": MODEL_ID,
            "device": 0 if use_cuda else -1,
            "torch_dtype": torch.float16 if use_cuda else torch.float32,
        }

        if MODEL_REVISION:
            kwargs["revision"] = MODEL_REVISION

        if CHUNK_LENGTH_S > 0:
            kwargs["chunk_length_s"] = CHUNK_LENGTH_S

        print(
            f"[STT] Loading model={MODEL_ID}, "
            f"device={'cuda:0' if use_cuda else 'cpu'}, "
            f"chunk_length_s={CHUNK_LENGTH_S}",
            flush=True,
        )

        t0 = time.perf_counter()
        _asr = pipeline("automatic-speech-recognition", **kwargs)
        print(
            f"[STT] Model load: {time.perf_counter() - t0:.3f}s",
            flush=True,
        )

    return _asr


def handler(job):
    data = job.get("input") or {}
    task = data.get("task", "stt")

    if task == "health":
        if data.get("load", False):
            load_model()

        return {
            "status": "ok",
            "model": MODEL_ID,
            "loaded": _asr is not None,
            "cuda_available": torch.cuda.is_available(),
            "gpu": (
                torch.cuda.get_device_name(0)
                if torch.cuda.is_available()
                else None
            ),
        }

    if task != "stt":
        raise ValueError("Supported tasks: health, stt")

    audio_b64 = data.get("audio_b64")
    if not audio_b64:
        raise ValueError("Provide audio_b64 containing a WAV file")

    total_start = time.perf_counter()

    try:
        t0 = time.perf_counter()
        audio_bytes = base64.b64decode(audio_b64, validate=True)
        audio, sample_rate = sf.read(
            io.BytesIO(audio_bytes),
            dtype="float32",
        )

        if audio.size == 0:
            raise ValueError("The audio file is empty")

        if audio.ndim == 2:
            audio = audio.mean(axis=1)

        if sample_rate != 16000:
            audio = resample_poly(
                audio, 16000, sample_rate
            ).astype(np.float32)
            sample_rate = 16000

        audio = np.ascontiguousarray(audio, dtype=np.float32)
        preprocess_s = time.perf_counter() - t0

        t0 = time.perf_counter()
        asr = load_model()
        load_or_retrieve_s = time.perf_counter() - t0

        # Ensure GPU execution time is measured accurately.
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        t0 = time.perf_counter()
        result = asr({
            "raw": audio,
            "sampling_rate": sample_rate,
        })

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        inference_s = time.perf_counter() - t0

        text = result["text"].strip()
        total_s = time.perf_counter() - total_start

        print(
            f"[STT_TIMING] "
            f"audio_s={len(audio) / sample_rate:.2f} "
            f"preprocess_s={preprocess_s:.3f} "
            f"load_or_retrieve_s={load_or_retrieve_s:.3f} "
            f"inference_s={inference_s:.3f} "
            f"total_s={total_s:.3f} "
            f"cuda={torch.cuda.is_available()} "
            f"text_chars={len(text)}",
            flush=True,
        )

        return {
            "text": text,
            "model": MODEL_ID,
        }

    except Exception as exc:
        print(f"STT inference error: {exc}", flush=True)
        traceback.print_exc()
        raise


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
