"""Unified local Ollama LLM client with explicit timeout, token budgets, and JSON mode."""
import logging
import os
import time
from typing import Optional

import requests

from src.config import LLM_OLLAMA_TIMEOUT_S
from src.judge_models import is_thinking_model

logger = logging.getLogger(__name__)

OLLAMA_TIMEOUT_S = LLM_OLLAMA_TIMEOUT_S


class UnifiedLLMClient:
    """Local Ollama LLM client for both generation and multi-judge evaluation."""

    def __init__(
        self,
        model: Optional[str] = None,
        provider: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 512,
        strict: bool = False,
        is_judge: Optional[bool] = None,
    ):
        from src.config import (
            JUDGE_HF_MODEL,
            JUDGE_OLLAMA_MODEL,
            LLM_HF_MODEL,
            LLM_MODEL,
            LLM_MODEL_ALIASES,
            LLM_PROVIDER,
        )

        self.strict = strict
        self.is_judge = bool(is_judge) if is_judge is not None else (
            model == JUDGE_HF_MODEL or model == "judge"
        )
        self.last_usage: dict = {}

        requested_provider = (provider or os.getenv("LLM_PROVIDER") or LLM_PROVIDER).strip().lower()
        if requested_provider and requested_provider != "ollama":
            if strict:
                raise RuntimeError(
                    f"Remote LLM provider {requested_provider!r} is disabled (local Ollama only)."
                )
            logger.warning("Remote provider %r disabled; using local Ollama.", requested_provider)

        self.provider = "ollama"
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.ollama_model = (
            JUDGE_OLLAMA_MODEL
            if model in (JUDGE_HF_MODEL, "judge")
            else LLM_MODEL
            if model is None or model in (LLM_MODEL, LLM_HF_MODEL, *LLM_MODEL_ALIASES)
            else model
        )
        self.model = self.ollama_model

    def _ollama_options(self, temp: float, tokens: int) -> dict:
        options = {"temperature": temp, "num_predict": tokens}
        if self.is_judge:
            # Pairwise prompts carry two answers + two evidence blocks; the
            # small Ollama default context silently drops the instructions.
            from src.config import JUDGE_NUM_CTX
            options["num_ctx"] = JUDGE_NUM_CTX
        return options

    @staticmethod
    def _ollama_base_url() -> str:
        host = (os.getenv("OLLAMA_HOST") or "http://localhost:11434").strip().rstrip("/")
        if not host.startswith(("http://", "https://")):
            host = f"http://{host}"
        return host

    def _chat_ollama(self, messages, temp, tokens, json_mode, model):
        import ollama
        kwargs = {"options": self._ollama_options(temp, tokens)}
        keep_alive = os.getenv("OLLAMA_KEEP_ALIVE", "").strip()
        if keep_alive:
            kwargs["keep_alive"] = int(keep_alive) if keep_alive.lstrip("-").isdigit() else keep_alive
        if json_mode:
            kwargs["format"] = "json"
        if is_thinking_model(model):
            kwargs["think"] = False
        _client_cls = getattr(ollama, "Client", None)
        _chat = (_client_cls(timeout=OLLAMA_TIMEOUT_S).chat
                 if _client_cls is not None else ollama.chat)
        try:
            res = _chat(model=model, messages=messages, **kwargs)
        except TypeError:
            kwargs.pop("think", None)
            kwargs.pop("keep_alive", None)
            res = _chat(model=model, messages=messages, **kwargs)
        raw = res if isinstance(res, dict) else getattr(res, "__dict__", {})
        self.last_usage = _ollama_usage(raw)
        if isinstance(res, dict):
            return (res.get("message") or {}).get("content") or ""
        return getattr(getattr(res, "message", None), "content", None) or ""

    def _chat_ollama_http(self, messages, temp, tokens, json_mode, model):
        """Timeout-guarded HTTP call (the ollama pkg has no timeout kwarg)."""
        payload = {
            "model": model,
            "messages": messages,
            "stream": False,
            "options": self._ollama_options(temp, tokens),
        }
        keep_alive = os.getenv("OLLAMA_KEEP_ALIVE", "").strip()
        if keep_alive:
            payload["keep_alive"] = int(keep_alive) if keep_alive.lstrip("-").isdigit() else keep_alive
        if json_mode:
            payload["format"] = "json"
        if is_thinking_model(model):
            # qwen3 / deepseek-r1: without this the reply starts with a long
            # <think> block that burns num_predict and breaks JSON parsing.
            payload["think"] = False
        try:
            resp = requests.post(
                f"{self._ollama_base_url()}/api/chat", json=payload, timeout=OLLAMA_TIMEOUT_S,
            )
            resp.raise_for_status()
            data = resp.json()
            self.last_usage = _ollama_usage(data)
            return data.get("message", {}).get("content", "") or ""
        except requests.Timeout as e:
            raise RuntimeError(
                f"Ollama call to {model!r} timed out after {OLLAMA_TIMEOUT_S:.0f} s"
            ) from e
        except Exception as e:
            logger.warning("Ollama HTTP call failed (%s); retrying via SDK.", e)
            return self._chat_ollama(messages, temp, tokens, json_mode, model)

    def check_ready(self) -> None:
        """Fail fast if Ollama is unreachable, model is not pulled, or returns empty."""
        try:
            resp = requests.get(f"{self._ollama_base_url()}/api/tags", timeout=10)
            resp.raise_for_status()
            installed = {m.get("name") for m in resp.json().get("models", [])}
        except Exception as e:
            raise RuntimeError(
                f"Ollama server is not reachable on {self._ollama_base_url()} ({e}). "
                "Start it and retry."
            ) from e

        from src.config import LLM_MODEL, LLM_MODEL_ALIASES
        wanted = {self.model, f"{self.model}:latest"}
        if self.model == LLM_MODEL:
            for alias in LLM_MODEL_ALIASES:
                wanted.update((alias, f"{alias}:latest"))
        if not wanted & installed:
            raise RuntimeError(
                f"Model {self.model!r} is not pulled in Ollama. Run "
                f"`ollama pull {self.model}` (installed: "
                f"{', '.join(sorted(n for n in installed if n)) or 'none'})."
            )
        if self.model not in installed and f"{self.model}:latest" not in installed:
            for alias in LLM_MODEL_ALIASES:
                for tag in (alias, f"{alias}:latest"):
                    if tag in installed:
                        self.model = tag
                        break
                if self.model in installed:
                    break

        try:
            reply = self.chat(
                [{"role": "user", "content": 'Reply with exactly this JSON: {"ok": true}'}],
                temperature=0.0, max_tokens=512, json_mode=True,
            )
        except Exception as e:
            raise RuntimeError(
                f"ollama model {self.model!r} failed a test call: {e}"
            ) from e
        if not (reply or "").strip():
            raise RuntimeError(
                f"ollama model {self.model!r} returned an empty reply "
                "to a test call (out of memory, or not loaded?). Check "
                "`ollama ps` and /tmp/ollama.log."
            )

    def chat(
        self,
        messages: list[dict],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        json_mode: bool = False,
    ) -> str:
        """Execute local Ollama chat completion."""
        temp = temperature if temperature is not None else self.temperature
        tokens = max_tokens if max_tokens is not None else self.max_tokens
        self.last_usage = {}
        return self._chat_ollama_http(messages, temp, tokens, json_mode, self.model)

    def chat_with_stats(
        self,
        messages: list[dict],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        json_mode: bool = False,
    ) -> tuple[str, dict]:
        """chat() plus generation-cost accounting for the latency breakdown."""
        t0 = time.perf_counter()
        text = self.chat(messages, temperature=temperature,
                         max_tokens=max_tokens, json_mode=json_mode)
        wall = time.perf_counter() - t0
        usage = dict(self.last_usage or {})
        completion = usage.get("completion_tokens")
        decode_s = usage.get("decode_s")
        if completion and decode_s:
            tps = completion / decode_s
        elif completion and wall > 0:
            tps = completion / wall
        else:
            tps = None
        stats = {
            "t_generation": wall,
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": completion,
            "prefill_s": usage.get("prefill_s"),
            "decode_s": decode_s,
            "tokens_per_second": tps,
            "provider": self.provider,
            "model": self.model,
        }
        return text, stats


def _ollama_usage(data: dict) -> dict:
    """Token/timing counters from an Ollama /api/chat response (ns -> s)."""
    data = data or {}

    def _sec(key):
        val = data.get(key)
        return val / 1e9 if isinstance(val, (int, float)) else None

    return {
        "prompt_tokens": data.get("prompt_eval_count"),
        "completion_tokens": data.get("eval_count"),
        "prefill_s": _sec("prompt_eval_duration"),
        "decode_s": _sec("eval_duration"),
    }
