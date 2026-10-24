FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime

WORKDIR /app

ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV HF_HOME=/app/hf_cache
ENV MODEL_ID=mmeog/mme_stt_model

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Bake model weights into the image so cold starts never download from Hugging Face.
# If the model repo is PRIVATE, build with:  --secret id=hf_token,src=hf_token.txt
# (or in GitHub Actions pass a build secret named hf_token) and use the line below instead:
#   RUN --mount=type=secret,id=hf_token \
#       HF_TOKEN=$(cat /run/secrets/hf_token) python -c "from huggingface_hub import snapshot_download; snapshot_download('mmeog/mme_stt_model')"
RUN python -c "from huggingface_hub import snapshot_download; snapshot_download('mmeog/mme_stt_model')"

# Weights are local now: skip Hub network checks at startup
ENV HF_HUB_OFFLINE=1

COPY handler.py .

CMD ["python", "-u", "handler.py"]
