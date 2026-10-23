
import base64
import io
import os
import threading
import traceback

import numpy as np
import runpod
import soundfile as sf
import torch
from scipy.signal import resample_poly
from transformers import pipeline

MODEL_ID = os.getenv("MODEL_ID", "mmeog/mme_stt_model")
MODEL_REVISION = os.getenv("MODEL_REVISION", "")

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

        _asr = pipeline(
            "automatic-speech-recognition",
            **kwargs,
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
        }

    if task != "stt":
        raise ValueError("Supported tasks: health, stt")

    audio_b64 = data.get("audio_b64")
    if not audio_b64:
        raise ValueError("Provide audio_b64 containing a WAV file")

    try:
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
                audio,
                16000,
                sample_rate,
            ).astype(np.float32)
            sample_rate = 16000

        audio = np.asarray(audio, dtype=np.float32)

        asr = load_model()
        result = asr({
            "raw": audio,
            "sampling_rate": sample_rate,
        })

        return {
            "text": result["text"].strip(),
            "model": MODEL_ID,
        }

    except Exception as exc:
        print(f"STT inference error: {exc}")
        traceback.print_exc()
        raise


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
