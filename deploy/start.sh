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
nohup ollama serve > /data/ollama.log 2>&1 &
for _ in $(seq 1 30); do
    if curl -sf http://127.0.0.1:11434/api/tags >/dev/null 2>&1; then
        break
    fi
    sleep 1
done
# 4-bit tags (same as src/config.LLM_MODEL and judge registry index 0)
ollama pull llama3.1:8b-instruct-q4_K_M || echo "model pull failed - check volume"
ollama pull qwen3:14b-q4_K_M || true
service ssh start || /usr/sbin/sshd
tail -f /data/ollama.log
