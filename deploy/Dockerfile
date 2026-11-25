# HCAD-RAG on NVIDIA A100 (Akash / any CUDA cloud)
FROM nvidia/cuda:12.6.3-cudnn-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1
RUN apt-get update && apt-get install -y python3.12 python3-pip git curl lsof openssh-server zstd \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
COPY requirements.txt .
# GPU torch: CUDA 12.6 wheels match A100 provider drivers; preserve numpy==2.5.4 + SciSpacy NER pins
RUN pip3 install --break-system-packages torch --index-url https://download.pytorch.org/whl/cu126 \
    && pip3 install --break-system-packages -r requirements.txt \
    && pip3 install --break-system-packages --force-reinstall --no-deps numpy==2.5.4 \
    && pip3 install --break-system-packages --no-deps \
        https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_ner_bc5cdr_md-0.5.4.tar.gz \
        https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_ner_jnlpba_md-0.5.4.tar.gz
COPY . /workspace/HCAD-RAG
WORKDIR /workspace/HCAD-RAG

# Ollama runs inside the container; models live on the persistent volume
ENV OLLAMA_MODELS=/data/ollama
COPY deploy/start.sh /start.sh
RUN chmod +x /start.sh
EXPOSE 22 8000 11434
ENTRYPOINT ["/start.sh"]
