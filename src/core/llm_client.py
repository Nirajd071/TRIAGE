"""Unified LLM client (audit fixes: no stateful provider mutation, Ollama timeout,
honest token budgets, explicit JSON mode)."""
import logging
import os
import time
from typing import Optional

import requests

from src.config import LLM_OLLAMA_TIMEOUT_S
from src.judge_models import is_thinking_model

logger = logging.getLogger(__name__)

OLLAMA_TIMEOUT_S = LLM_OLLAMA_TIMEOUT_S


def _retry_wait_seconds(resp: Optional[requests.Response], attempt: int, base: float = 3.0) -> float:
    """Compute exponential backoff wait time, honoring HTTP 429 Retry-After."""
    wait_s = min(60.0, base * (2 ** attempt))
    if resp is not None and getattr(resp, "status_code", None) == 429:
        retry_hdr = resp.headers.get("retry-after") or resp.headers.get("Retry-After")
        if retry_hdr:
            try:
                wait_s = max(wait_s, min(65.0, float(retry_hdr) + 1.0))
            except ValueError:
                pass
    return wait_s

class UnifiedLLMClient:
    """Unified LLM interface supporting Hugging Face Inference API with fallback.

    Fixes applied :
    - A transient HF failure no longer permanently mutates the instance; the
      fallback is scoped to the failing call.
    - Ollama calls carry an explicit timeout.
    - `max_tokens` is honored (no silent 768 cap).
    - JSON mode is an explicit parameter, not substring sniffing.
    """

    def __init__(
        self,
        model: Optional[str] = None,
        provider: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 512,
        strict: bool = False,
        is_judge: Optional[bool] = None,
    ):
        """
        Args:
            strict: when True the client NEVER substitutes another provider
                or model (no HF/NIM -> Ollama fallback). Judges of the
                multi-judge ensemble are strict, so a missing API key or a
                failed call surfaces as an error instead of silently scoring
                with a different model under the wrong judge name.
            is_judge: marks a judge client (larger num_ctx). Defaults to True
                for the legacy JUDGE_HF_MODEL / "judge" aliases.
        """
        from src.config import (
            LLM_PROVIDER, LLM_HF_MODEL, JUDGE_HF_MODEL, JUDGE_OLLAMA_MODEL,
            LLM_MODEL, LLM_MODEL_ALIASES, HF_TOKEN,
        )

        self.strict = strict
        self.is_judge = bool(is_judge) if is_judge is not None else (
            model == JUDGE_HF_MODEL or model == "judge")
        self.last_usage: dict = {}
        self.provider = (provider or os.getenv("LLM_PROVIDER") or LLM_PROVIDER).lower()
        # Judge-specific provider override: run the generator fully local
        # (e.g., ollama on a small GPU) while the judge uses a stronger
        # remote model (e.g., huggingface cloud = Qwen2.5-72B-Instruct).
        # An explicit provider kwarg always wins over the env override.
        if not provider and (model == JUDGE_HF_MODEL or model == "judge"):
            judge_provider = (os.getenv("JUDGE_PROVIDER") or "").strip().lower()
            if judge_provider:
                self.provider = judge_provider
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.token = os.getenv("HF_TOKEN") or HF_TOKEN
        self.ollama_model = (
            JUDGE_OLLAMA_MODEL
            if model in (JUDGE_HF_MODEL, "judge")
            else LLM_MODEL
            if model is None or model in (LLM_MODEL, LLM_HF_MODEL, *LLM_MODEL_ALIASES)
            else model
        )

        if self.provider == "nvidia":
            from src.config import NVIDIA_API_KEY, NVIDIA_BASE_URL, LLM_NVIDIA_MODEL, JUDGE_NVIDIA_MODEL
            self.nvidia_key = os.getenv("NVIDIA_API_KEY") or NVIDIA_API_KEY
            self.nvidia_url = (os.getenv("NVIDIA_BASE_URL") or NVIDIA_BASE_URL).rstrip("/") + "/chat/completions"
            if model == JUDGE_HF_MODEL or model == "judge":
                self.model = JUDGE_NVIDIA_MODEL
            elif not model or model in (LLM_MODEL, LLM_HF_MODEL, *LLM_MODEL_ALIASES):
                self.model = LLM_NVIDIA_MODEL
            else:
                self.model = model
            if not self.nvidia_key:
                if strict:
                    raise RuntimeError(
                        f"NVIDIA_API_KEY is not set but strict client requested "
                        f"nvidia:{self.model}")
                logger.warning("NVIDIA_API_KEY not found; falling back to Ollama.")
                self.provider = "ollama"
                self.model = self.ollama_model
            else:
                return

        if self.provider == "gemini":
            from src.config import GEMINI_API_KEY
            self.gemini_key = os.getenv("GEMINI_API_KEY") or GEMINI_API_KEY
            self.model = model if (model and model not in (JUDGE_HF_MODEL, "judge", LLM_MODEL)) else os.getenv("GEMINI_JUDGE_MODEL", "gemini-flash-lite-latest")
            if not self.gemini_key:
                if strict:
                    raise RuntimeError(f"GEMINI_API_KEY is not set but strict client requested gemini:{self.model}")
                logger.warning("GEMINI_API_KEY not found; falling back to Ollama.")
                self.provider = "ollama"
                self.model = self.ollama_model
            else:
                return

        if self.provider == "groq":
            from src.config import GROQ_API_KEY
            self.groq_key = os.getenv("GROQ_API_KEY") or GROQ_API_KEY
            self.model = model if (model and model not in (JUDGE_HF_MODEL, "judge", LLM_MODEL)) else "qwen/qwen3.8-27b"
            if not self.groq_key:
                if strict:
                    raise RuntimeError(f"GROQ_API_KEY is not set but strict client requested groq:{self.model}")
                logger.warning("GROQ_API_KEY not found; falling back to Ollama.")
                self.provider = "ollama"
                self.model = self.ollama_model
            else:
                return

        if self.provider == "openrouter":
            from src.config import OPENROUTER_API_KEY
            self.openrouter_key = os.getenv("OPENROUTER_API_KEY") or OPENROUTER_API_KEY
            self.model = model if (model and model not in (JUDGE_HF_MODEL, "judge", LLM_MODEL)) else "nvidia/nemotron-3.5-lightning:free"
            if not self.openrouter_key:
                if strict:
                    raise RuntimeError(f"OPENROUTER_API_KEY is not set but strict client requested openrouter:{self.model}")
                logger.warning("OPENROUTER_API_KEY not found; falling back to Ollama.")
                self.provider = "ollama"
                self.model = self.ollama_model
            else:
                return

        if self.provider == "huggingface" and self.token:
            self.model = LLM_HF_MODEL if (not model or model == LLM_MODEL) else model
            try:
                from huggingface_hub import InferenceClient
                self._hf_client = InferenceClient(api_key=self.token, timeout=90.0)
                return
            except Exception as e:
                if strict:
                    raise RuntimeError(f"HF client unavailable for strict client: {e}") from e
                logger.warning("Failed to initialize HF client: %s. Using Ollama.", e)
        else:
            if strict and self.provider == "huggingface":
                raise RuntimeError("HF_TOKEN is not set but strict client requested huggingface")
            if self.provider != "nvidia":
                logger.info("HF provider unavailable (no token) - using Ollama.")
        self.provider = "ollama"
        self.model = self.ollama_model

    def _ollama_options(self, temp, tokens):
        options = {"temperature": temp, "num_predict": tokens}
        if self.is_judge:
            # Pairwise prompts carry two answers + two evidence blocks; the
            # small Ollama default context silently drops the instructions.
            from src.config import JUDGE_NUM_CTX
            options["num_ctx"] = JUDGE_NUM_CTX
        return options

    def _chat_openai_compat(
        self,
        label: str,
        url: str,
        api_key: str,
        messages: list[dict],
        temp: float,
        tokens: int,
        json_mode: bool,
        ollama_target: str,
    ) -> str:
        """Call an OpenAI-compatible chat completions endpoint with backoff and 429 Retry-After."""
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temp,
            "max_tokens": tokens,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        for attempt in range(5):
            resp = None
            try:
                resp = requests.post(url, json=payload, headers=headers, timeout=60.0)
                resp.raise_for_status()
                data = resp.json()
                usage = data.get("usage") or {}
                self.last_usage = {
                    "prompt_tokens": usage.get("prompt_tokens"),
                    "completion_tokens": usage.get("completion_tokens"),
                }
                return data["choices"][0]["message"]["content"] or ""
            except Exception as e:
                logger.warning("%s attempt %d failed: %s", label, attempt + 1, e)
                if attempt < 4:
                    time.sleep(_retry_wait_seconds(resp, attempt))
                elif self.strict:
                    raise RuntimeError(f"{label} call failed for strict client {self.model}: {e}") from e
                else:
                    logger.warning("%s failed; falling back to Ollama (%s).", label, ollama_target)
                    return self._chat_ollama_http(messages, temp, tokens, json_mode, ollama_target)
        return ""

    def _chat_ollama(self, messages, temp, tokens, json_mode, model):
        import ollama
        kwargs = {"options": self._ollama_options(temp, tokens)}
        if json_mode:
            kwargs["format"] = "json"
        if is_thinking_model(model):
            kwargs["think"] = False
        # Bounded call: the module-level ollama.chat has no timeout at all.
        _client_cls = getattr(ollama, "Client", None)
        _chat = (_client_cls(timeout=OLLAMA_TIMEOUT_S).chat
                 if _client_cls is not None else ollama.chat)
        try:
            res = _chat(model=model, messages=messages, **kwargs)
        except TypeError:
            kwargs.pop("think", None)
            res = _chat(model=model, messages=messages, **kwargs)
        raw = res if isinstance(res, dict) else getattr(res, "__dict__", {})
        self.last_usage = _ollama_usage(raw)
        return res.message.content or ""

    def _chat_ollama_http(self, messages, temp, tokens, json_mode, model):
        """Timeout-guarded HTTP call (the ollama pkg has no timeout kwarg)."""
        payload = {
            "model": model, "messages": messages, "stream": False,
            "options": self._ollama_options(temp, tokens),
        }
        if json_mode:
            payload["format"] = "json"
        if is_thinking_model(model):
            # qwen3 / deepseek-r1: without this the reply starts with a long
            # <think> block that burns num_predict and breaks JSON parsing.
            payload["think"] = False
        try:
            resp = requests.post(
                "http://localhost:11434/api/chat", json=payload, timeout=OLLAMA_TIMEOUT_S,
            )
            resp.raise_for_status()
            data = resp.json()
            self.last_usage = _ollama_usage(data)
            return data.get("message", {}).get("content", "") or ""
        except requests.Timeout as e:
            # The server is still busy with this request (NUM_PARALLEL=1);
            # re-sending it via the SDK would queue the same work twice.
            raise RuntimeError(
                f"Ollama call to {model!r} timed out after {OLLAMA_TIMEOUT_S:.0f} s"
            ) from e
        except Exception as e:
            logger.warning("Ollama HTTP call failed (%s); retrying via SDK.", e)
            return self._chat_ollama(messages, temp, tokens, json_mode, model)

    def check_ready(self) -> None:
        """Fail fast if this client cannot answer (benchmark preflight).

        Without it, a judge that is not pulled or does not fit in memory
        only shows up later as every judge call failing. Raises
        RuntimeError with an actionable message.
        """
        if self.provider == "ollama":
            try:
                resp = requests.get("http://localhost:11434/api/tags", timeout=10)
                resp.raise_for_status()
                installed = {m.get("name") for m in resp.json().get("models", [])}
            except Exception as e:
                raise RuntimeError(
                    f"Ollama server is not reachable on localhost:11434 ({e}). "
                    "Start it (notebook: re-run the ollama cell) and retry."
                ) from e
            # Ollama reports untagged pulls as "<name>:latest".
            wanted = {self.model, f"{self.model}:latest"}
            if not wanted & installed:
                raise RuntimeError(
                    f"Model {self.model!r} is not pulled in Ollama. Run "
                    f"`ollama pull {self.model}` (installed: "
                    f"{', '.join(sorted(n for n in installed if n)) or 'none'})."
                )
        try:
            reply = self.chat(
                [{"role": "user", "content": 'Reply with exactly this JSON: {"ok": true}'}],
                temperature=0.0, max_tokens=32, json_mode=True,
            )
        except Exception as e:
            raise RuntimeError(
                f"{self.provider} model {self.model!r} failed a test call: {e}"
            ) from e
        if not (reply or "").strip():
            raise RuntimeError(
                f"{self.provider} model {self.model!r} returned an empty reply "
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
        """Execute chat completion across Hugging Face or Ollama.

        Provider fallback is scoped to this call; the instance state is never
        mutated, so a judge client cannot silently degrade mid-benchmark.
        """
        temp = temperature if temperature is not None else self.temperature
        tokens = max_tokens if max_tokens is not None else self.max_tokens
        self.last_usage = {}

        ollama_target = self.model if self.provider == "ollama" else self.ollama_model

        if self.provider == "nvidia":
            payload = {
                "model": self.model,
                "messages": messages,
                "temperature": temp,
                "max_tokens": tokens,
            }
            if json_mode:
                payload["response_format"] = {"type": "json_object"}
            headers = {
                "Authorization": f"Bearer {self.nvidia_key}",
                "Content-Type": "application/json",
            }
            try:
                resp = requests.post(self.nvidia_url, json=payload, headers=headers, timeout=60.0)
                resp.raise_for_status()
                data = resp.json()
                usage = data.get("usage") or {}
                self.last_usage = {
                    "prompt_tokens": usage.get("prompt_tokens"),
                    "completion_tokens": usage.get("completion_tokens"),
                }
                return data["choices"][0]["message"]["content"] or ""
            except Exception as e:
                if self.strict:
                    raise RuntimeError(
                        f"NVIDIA NIM call failed for strict client {self.model}: {e}") from e
                logger.warning("NVIDIA NIM call failed (%s); falling back to Ollama.", e)

        if self.provider == "gemini":
            candidate_models = [self.model]
            if not self.strict and ("flash-lite" in self.model or "flash" in self.model):
                for alt in ["gemini-flash-lite-latest", "gemini-3.5-flash-lite"]:
                    if alt not in candidate_models:
                        candidate_models.append(alt)

            headers = {
                "X-goog-api-key": self.gemini_key,
                "Content-Type": "application/json",
            }
            contents = []
            system_instruction = None
            for m in messages:
                role = m.get("role", "user")
                text = m.get("content", "")
                if role == "system":
                    system_instruction = {"parts": [{"text": text}]}
                else:
                    g_role = "user" if role == "user" else "model"
                    contents.append({"role": g_role, "parts": [{"text": text}]})
            if not contents and system_instruction:
                contents = [{"role": "user", "parts": system_instruction["parts"]}]
                system_instruction = None

            generation_config = {
                "temperature": temp,
                "maxOutputTokens": tokens,
            }
            if json_mode:
                generation_config["responseMimeType"] = "application/json"

            payload = {"contents": contents, "generationConfig": generation_config}
            if system_instruction:
                payload["systemInstruction"] = system_instruction

            last_err = None
            for cur_model in candidate_models:
                gemini_url = f"https://generativelanguage.googleapis.com/v1beta/models/{cur_model}:generateContent"
                for attempt in range(5):
                    try:
                        resp = requests.post(gemini_url, json=payload, headers=headers, timeout=60.0)
                        resp.raise_for_status()
                        data = resp.json()
                        usage = data.get("usageMetadata") or {}
                        self.last_usage = {
                            "prompt_tokens": usage.get("promptTokenCount"),
                            "completion_tokens": usage.get("candidatesTokenCount"),
                        }
                        candidates = data.get("candidates") or []
                        if candidates and "content" in candidates[0]:
                            parts = candidates[0]["content"].get("parts") or []
                            if parts:
                                return parts[0].get("text") or ""
                        return ""
                    except Exception as e:
                        last_err = e
                        logger.warning("Gemini (%s) attempt %d failed: %s", cur_model, attempt + 1, e)
                        if attempt < 4:
                            time.sleep(min(16.0, 2.0 * (2 ** attempt)))
                logger.warning("Gemini model %s exhausted retries; trying alternate candidate if available.", cur_model)

            if self.strict:
                raise RuntimeError(f"Gemini call failed for strict client {self.model}: {last_err}") from last_err
            else:
                logger.warning("Gemini failed; falling back to Ollama (%s).", ollama_target)
                return self._chat_ollama_http(messages, temp, tokens, json_mode, ollama_target)

        if self.provider == "groq":
            return self._chat_openai_compat(
                "Groq", "https://api.groq.com/openai/v1/chat/completions",
                self.groq_key, messages, temp, tokens, json_mode, ollama_target,
            )

        if self.provider == "openrouter":
            return self._chat_openai_compat(
                "OpenRouter", "https://openrouter.ai/api/v1/chat/completions",
                self.openrouter_key, messages, temp, tokens, json_mode, ollama_target,
            )

        if self.provider == "huggingface":
            for attempt in range(2):
                try:
                    hf_kwargs = {}
                    if json_mode:
                        hf_kwargs["response_format"] = {"type": "json_object"}
                    try:
                        res = self._hf_client.chat.completions.create(
                            model=self.model, messages=messages,
                            max_tokens=tokens, temperature=temp, **hf_kwargs,
                        )
                    except TypeError:
                        res = self._hf_client.chat.completions.create(
                            model=self.model, messages=messages,
                            max_tokens=tokens, temperature=temp,
                        )
                    usage = getattr(res, "usage", None)
                    if usage is not None:
                        self.last_usage = {
                            "prompt_tokens": getattr(usage, "prompt_tokens", None),
                            "completion_tokens": getattr(usage, "completion_tokens", None),
                        }
                    return res.choices[0].message.content or ""
                except Exception as e:
                    err = str(e)
                    logger.warning("HF attempt %d failed: %s", attempt + 1, e)
                    if "402" in err or "Payment Required" in err or "depleted" in err:
                        break
                    if attempt == 0:
                        time.sleep(1.0)
            if self.strict:
                raise RuntimeError(f"HF call failed for strict client {self.model}")
            logger.warning("Falling back to Ollama (%s) for this call only.", ollama_target)

        return self._chat_ollama_http(messages, temp, tokens, json_mode, ollama_target)

    def chat_with_stats(
        self,
        messages: list[dict],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        json_mode: bool = False,
    ) -> tuple[str, dict]:
        """chat() plus generation-cost accounting for the latency breakdown.

        Returns (text, stats): wall-clock ``t_generation`` (prefill + decode,
        network included), prompt / generated token counts reported by the
        provider, and decode tokens-per-second. Ollama additionally reports
        server-side ``prefill_s`` / ``decode_s``. Token counts are None when
        a provider does not report usage.
        """
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
