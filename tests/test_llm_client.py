"""Offline tests for the shared LLM client against a fake Ollama-compatible server."""
from __future__ import annotations

import asyncio
import json
import socket

import pytest
from aiohttp import web


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture()
def fake_ollama(monkeypatch):
    calls: list[dict] = []

    async def chat(request):
        body = await request.json()
        calls.append(body)
        content = "<think>private reasoning</think>\n<thinking>kept</thinking>\nKhuyến nghị: MUA"
        return web.json_response({"choices": [{"message": {"role": "assistant", "content": content}}]})

    port = _free_port()
    from ta_core.core import llm

    monkeypatch.setattr(llm, "BASE_URL", f"http://127.0.0.1:{port}/v1")

    async def start():
        app = web.Application()
        app.router.add_post("/v1/chat/completions", chat)
        runner = web.AppRunner(app)
        await runner.setup()
        await web.TCPSite(runner, "127.0.0.1", port).start()
        return runner

    return llm, calls, start


def test_defaults_point_at_local_ollama(monkeypatch):
    for name in ("LLM_BASE_URL", "CLIPROXY_BASE_URL", "LLM_API_KEY", "CLIPROXY_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    import config

    assert config._llm_base_url() == "http://127.0.0.1:11434/v1"
    assert config._llm_api_key() == "ollama"


def test_reasoning_model_used_for_debate_and_cio(monkeypatch):
    for name in ("T3_DEBATE", "T3_DEBATE_PRIMARY", "T4_CIO", "T4_CIO_PRIMARY", "T2_TECHNICAL", "T2_TECHNICAL_PRIMARY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("LLM_MODEL", "qwen3:8b")
    monkeypatch.setenv("LLM_REASONING_MODEL", "qwen3:14b")
    import config

    assert config._model("T2_TECHNICAL", "T2_TECHNICAL_PRIMARY") == "qwen3:8b"
    assert config._model("T4_CIO", "T4_CIO_PRIMARY", "LLM_REASONING_MODEL") == "qwen3:14b"


def test_think_block_is_stripped_and_thinking_tag_kept(fake_ollama):
    llm, calls, start = fake_ollama

    async def run():
        runner = await start()
        try:
            return await llm.call_llm("sys", "user", model="qwen3:8b")
        finally:
            await llm.close_session()
            await runner.cleanup()

    out = asyncio.run(run())
    assert "<think>" not in out and "private reasoning" not in out
    assert out.startswith("<thinking>kept</thinking>")
    assert calls[0]["model"] == "qwen3:8b"


def test_client_survives_multiple_event_loops(fake_ollama):
    """Backtests call asyncio.run() repeatedly; the cached session must not leak across loops."""
    llm, calls, start = fake_ollama

    async def once():
        runner = await start()
        try:
            return await llm.call_llm("sys", "user")
        finally:
            await runner.cleanup()

    assert asyncio.run(once()).endswith("Khuyến nghị: MUA")
    assert asyncio.run(once()).endswith("Khuyến nghị: MUA")
    assert len(calls) == 2
