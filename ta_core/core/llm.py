import asyncio
import atexit
import random
import re
from typing import Any

import aiohttp
from config import models

# One OpenAI-compatible endpoint for every agent; Ollama by default (see config.py).
API_KEY = models.llm_api_key
BASE_URL = models.llm_base_url
DEFAULT_MODEL = models.primary_model

# Reasoning models served by Ollama (Qwen3, DeepSeek-R1) put their reasoning in a
# <think>...</think> block before the answer. Agents parse the answer, so it is
# removed. The <thinking> tag some prompts ask for is a different tag and is kept.
_THINK_BLOCK = re.compile(r"<think>.*?</think>\s*", re.DOTALL)


def _strip_think(content: str) -> str:
    return _THINK_BLOCK.sub("", content).strip()

# ---------------------------------------------------------------------------
# Concurrency guard — limits in-flight requests (LLM_CONCURRENCY). A local
# Ollama server only runs a few requests in parallel (OLLAMA_NUM_PARALLEL).
# ---------------------------------------------------------------------------

_session = None
_session_loop = None
_sem = None
_sem_loop = None


def _get_semaphore() -> asyncio.Semaphore:
    """Per-event-loop semaphore: callers such as backtests run asyncio.run() repeatedly."""
    global _sem, _sem_loop
    loop = asyncio.get_running_loop()
    if _sem is None or _sem_loop is not loop:
        _sem = asyncio.Semaphore(models.llm_concurrency)
        _sem_loop = loop
    return _sem


async def _get_session():
    global _session, _session_loop
    loop = asyncio.get_running_loop()
    if _session is not None and not _session.closed and _session_loop is not loop:
        # Session belongs to an earlier, possibly closed, event loop; it cannot be reused.
        _session = None
    if _session is None or _session.closed:
        _session_loop = loop
        connector = aiohttp.TCPConnector(
            limit=models.llm_concurrency + 10,
            keepalive_timeout=720,
        )
        # Local inference on CPU can be slow, so the read timeout is generous.
        timeout = aiohttp.ClientTimeout(
            total=models.llm_timeout_seconds,
            connect=30,
            sock_read=models.llm_timeout_seconds,
        )
        _session = aiohttp.ClientSession(connector=connector, timeout=timeout)
    return _session


async def close_session():
    global _session
    if _session and not _session.closed:
        await _session.close()
        _session = None


def _cleanup_session_sync() -> None:
    global _session
    if _session and not _session.closed:
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(close_session())
        finally:
            loop.close()


atexit.register(_cleanup_session_sync)


# ---------------------------------------------------------------------------
# Retry helpers
# ---------------------------------------------------------------------------

_MAX_RETRIES_SAME_MODEL = 2
_INITIAL_BACKOFF_SECONDS = 1.5
_MAX_BACKOFF_SECONDS = 15.0


def _exp_backoff(attempt: int, initial: float = _INITIAL_BACKOFF_SECONDS) -> float:
    """Exponential backoff with jitter: ``initial * 2^attempt + rand``."""
    delay = min(initial * (2 ** attempt), _MAX_BACKOFF_SECONDS)
    return delay + random.uniform(0, 0.5)



class LLMError(Exception):
    """Raised when all LLM attempts fail; callers must handle explicitly."""


async def call_llm(
    system_prompt: str,
    user_prompt: str,
    model: str | None = None,
    temperature: float = 0.3,
    fallback_models: list[str] | None = None,
    response_format: dict[str, Any] | None = None,
) -> str:
    """
    * Raises LLMError instead of returning error strings.
    """
    del fallback_models
    url = f"{BASE_URL}/chat/completions"
    models_to_try = [model or DEFAULT_MODEL]

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}",
    }

    last_error = "❌ Lỗi kết nối LLM: Unknown error"

    for idx, chosen_model in enumerate(models_to_try):
        payload = {
            "model": chosen_model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": temperature,
            "stream": False,
        }
        if response_format is not None:
            payload["response_format"] = response_format

        # Retry loop for the SAME model (handles 429s / 5xx)
        for attempt in range(_MAX_RETRIES_SAME_MODEL):
            try:
                session = await _get_session()
                async with _get_semaphore():
                    async with session.post(url, headers=headers, json=payload) as response:
                        if response.status == 429:
                            retry_after = response.headers.get("Retry-After")
                            if retry_after:
                                try:
                                    wait = float(retry_after)
                                except ValueError:
                                    wait = _exp_backoff(attempt)
                            else:
                                wait = _exp_backoff(attempt)
                            print(
                                f"[call_llm] 429 on {chosen_model} "
                                f"(attempt {attempt + 1}/{_MAX_RETRIES_SAME_MODEL}). "
                                f"Waiting {wait:.1f}s …",
                                flush=True,
                            )
                            last_error = f"❌ Rate limited (429) on {chosen_model}"
                            await asyncio.sleep(wait)
                            continue  # retry same model

                        if response.status != 200:
                            text = await response.text()
                            last_error = f"❌ Lỗi API ({response.status}): {text}"
                            if response.status >= 500:
                                await asyncio.sleep(_exp_backoff(attempt, 1.0))
                                continue
                            break  # 4xx (not 429) — skip to next model

                        try:
                            data = await response.json()
                        except Exception as exc:
                            last_error = f"❌ Phản hồi API không đúng định dạng: {exc}"
                            break

                        if data and "choices" in data and len(data["choices"]) > 0:
                            content = data["choices"][0]["message"]["content"]
                            content = _strip_think(content if isinstance(content, str) else str(content))

                            if content.startswith("❌"):
                                last_error = content
                                if attempt < _MAX_RETRIES_SAME_MODEL - 1:
                                    await asyncio.sleep(_exp_backoff(attempt, 1.0))
                                    continue
                                break
                            return content
                        else:
                            last_error = f"❌ Phản hồi API không đúng định dạng: {data}"
                            break

            except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
                last_error = f"❌ Lỗi kết nối LLM: {exc}"
                if attempt < _MAX_RETRIES_SAME_MODEL - 1:
                    await asyncio.sleep(_exp_backoff(attempt, 1.0))
                    continue
            except Exception as exc:
                last_error = f"❌ Lỗi kết nối LLM: {exc}"
                break

    raise LLMError(last_error)
