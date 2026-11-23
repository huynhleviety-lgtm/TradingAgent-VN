"""Tests for the dashboard's system status checks (app/backend/services/system_service.py)."""
from __future__ import annotations

import asyncio
import importlib.util
import json
import sqlite3
import sys
from datetime import date, timedelta
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture()
def svc(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location(
        "system_service_under_test", ROOT / "app" / "backend" / "services" / "system_service.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    db = tmp_path / "vnstock.db"
    module._test_db = db
    monkeypatch.setattr(module, "paths", type("P", (), {
        "vnstock_db_path": db,
        "market_data_cache_path": tmp_path / "vmd.db",
    })())
    return module


def _seed(db: Path, last_days_ago: dict[str, int], with_qlib: bool = False) -> None:
    with sqlite3.connect(db) as con:
        con.execute("CREATE TABLE market_data_daily (ticker TEXT, date TEXT, close REAL)")
        for ticker, ago in last_days_ago.items():
            d = (date.today() - timedelta(days=ago)).isoformat() + " 00:00:00.000000"
            con.execute("INSERT INTO market_data_daily VALUES (?,?,?)", (ticker, d, 10.0))
        if with_qlib:
            con.execute(
                "CREATE TABLE qlib_scores (ticker TEXT, date TEXT, score REAL, rank_pct REAL, model TEXT,"
                " trained_until TEXT, test_rank_ic REAL, run_at TEXT)"
            )
            con.executemany(
                "INSERT INTO qlib_scores VALUES (?,?,?,?,?,?,?,?)",
                [("AAA", "2026-11-20", 0.02, 1.0, "m", "2026-03-01", 0.03, "x"),
                 ("BBB", "2026-11-20", -0.01, 0.5, "m", "2026-03-01", 0.03, "x"),
                 ("AAA", "2026-11-19", 0.05, 1.0, "m", "2026-03-01", 0.03, "x")],
            )
            con.execute("CREATE TABLE qlib_runs (run_at TEXT, model TEXT, meta TEXT)")
            con.execute("INSERT INTO qlib_runs VALUES (?,?,?)",
                        ("2026-11-20T00:00:00+00:00", "m", json.dumps({"rank_ic_mean": 0.03})))


def test_market_data_missing_database(svc):
    out = svc.market_data_status()
    assert out["ok"] is False and "crawl-vnstock" in out["message"]


def test_market_data_flags_stale_tickers(svc):
    _seed(svc._test_db, {"FPT": 0, "VCB": 10, "VNINDEX": 0})
    out = svc.market_data_status()
    assert out["stocks"] == 2 and out["indices"] == ["VNINDEX"]
    assert out["stale_tickers"] == ["VCB"] and out["ok"] is False


def test_market_data_fresh(svc):
    _seed(svc._test_db, {"FPT": 1, "VNINDEX": 1})
    assert svc.market_data_status()["ok"] is True


def test_qlib_scores_latest_day_best_first(svc):
    _seed(svc._test_db, {"FPT": 0}, with_qlib=True)
    out = svc.qlib_scores()
    assert out["date"] == "2026-11-20"
    assert [r["ticker"] for r in out["rows"]] == ["AAA", "BBB"]
    assert out["run"]["rank_ic_mean"] == pytest.approx(0.03)


def test_llm_unreachable_is_reported_not_raised(svc, monkeypatch):
    fake = type("M", (), {"llm_base_url": "http://127.0.0.1:9/v1", "llm_api_key": "x", "llm_concurrency": 1,
                          "t2_technical_model": "a", "t3_debate_model": "b", "t4_cio_model": "b", "news_model": "a"})()
    monkeypatch.setattr(svc, "models", fake)
    out = asyncio.run(svc.llm_status())
    assert out["reachable"] is False and out["ok"] is False and "Ollama" in out["message"]
