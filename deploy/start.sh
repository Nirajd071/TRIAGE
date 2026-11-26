#!/bin/bash
# Boots Ollama + SSH; models/data persist on the mounted volume (/data)
set -e
mkdir -p /data/ollama /root/.ssh /run/sshd
chmod 700 /root/.ssh
if [ -n "${SSH_PUBKEY:-}" ]; then
    printf '%s\n' "$SSH_PUBKEY" > /root/.ssh/authorized_keys
    chmod 600 /root/.ssh/authorized_keys
fi

if ! command -v ollama >/dev/null 2>&1; then
    curl -fsSL https://ollama.com/install.sh | sh
fi
export OLLAMA_FLASH_ATTENTION="${OLLAMA_FLASH_ATTENTION:-1}"
export OLLAMA_KV_CACHE_TYPE="${OLLAMA_KV_CACHE_TYPE:-q8_0}"
export OLLAMA_NUM_PARALLEL="${OLLAMA_NUM_PARALLEL:-1}"
export OLLAMA_MAX_LOADED_MODELS="${OLLAMA_MAX_LOADED_MODELS:-5}"
export OLLAMA_KEEP_ALIVE="${OLLAMA_KEEP_ALIVE:-60m}"
nohup ollama serve > /data/ollama.log 2>&1 &
for _ in $(seq 1 30); do
    if curl -sf http://127.0.0.1:11434/api/tags >/dev/null 2>&1; then
        break
    fi
    sleep 1
done
# 4-bit generator + 4-judge cross-vendor panel (src/config.LLM_MODEL + src/judge_models.py 0..3)
for tag in \
    llama3.1:8b-instruct-q4_K_M \
    qwen3:14b-q4_K_M \
    gemma3:12b-it-q4_K_M \
    phi4:14b-q4_K_M \
    mistral:7b-instruct-v0.3-q4_K_M
do
    ollama pull "$tag" || echo "model pull failed ($tag) - check volume"
done
service ssh start || /usr/sbin/sshd
tail -f /data/ollama.log
