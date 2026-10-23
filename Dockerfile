
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

WORKDIR /app

ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/runpod-volume/hf
ENV MODEL_ID=mmeog/mme_stt_model
ENV MODEL_REVISION=c31280c39562ad50d22f22a58e51cb1a4e233f71

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY handler.py .

CMD ["python", "-u", "handler.py"]
