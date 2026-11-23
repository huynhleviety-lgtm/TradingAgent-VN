"""AutoGen debate: the project's custom model client in a real RoundRobinGroupChat.

The LLM call is stubbed, so this needs no Ollama and no network.
"""
from __future__ import annotations

import asyncio

import pytest

pytest.importorskip("autogen_agentchat")


def test_groupchat_transcript_is_rendered(monkeypatch):
    import ta_core.workflows.debate.autogen_debate as debate
    from autogen_agentchat.agents import AssistantAgent
    from autogen_agentchat.teams import RoundRobinGroupChat

    calls: list[str] = []

    async def fake_llm(system, user, model=None, temperature=0.3, **kwargs):
        calls.append(model)
        return f"Luận điểm {len(calls)}"

    monkeypatch.setattr(debate, "call_llm", fake_llm)
    client = debate.VNStockModelClient(model="qwen3:14b")

    async def run():
        bull = AssistantAgent(name="Bull", model_client=client, system_message="bull")
        bear = AssistantAgent(name="Bear", model_client=client, system_message="bear")
        return await RoundRobinGroupChat([bull, bear], max_turns=4).run(task="FPT?")

    result = asyncio.run(run())
    transcript = debate._render_transcript(list(result.messages))

    assert calls == ["qwen3:14b"] * 4
    assert "[Bull] Luận điểm 1" in transcript
    assert "[Bear] Luận điểm 2" in transcript
    assert "[user]" not in transcript and "FPT?" not in transcript  # task prompt is not a debate turn
    assert not transcript.startswith("[DEBATE_FALLBACK]")
