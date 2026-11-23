"""Tests for reading Qlib scores, including the look-ahead guard."""
from __future__ import annotations

import sqlite3

import pytest

from ta_core.tools.qlib_scores import latest_score


@pytest.fixture()
def db(tmp_path):
    path = tmp_path / "vnstock.db"
    with sqlite3.connect(path) as con:
        con.execute(
            "CREATE TABLE qlib_scores (ticker TEXT, date TEXT, score REAL, rank_pct REAL, model TEXT,"
            " trained_until TEXT, test_rank_ic REAL, run_at TEXT)"
        )
        con.executemany(
            "INSERT INTO qlib_scores VALUES (?,?,?,?,?,?,?,?)",
            [
                ("FPT", "2026-11-18", 0.010, 0.70, "alpha158_lgbm", "2026-03-16", 0.03, "x"),
                ("FPT", "2026-11-20", 0.020, 0.90, "alpha158_lgbm", "2026-03-16", 0.03, "x"),
            ],
        )
    return path


def test_returns_latest_score_on_or_before_ref_date(db):
    s = latest_score("fpt", "2026-11-19", db_path=db)
    assert s["date"] == "2026-11-18" and s["rank_pct"] == pytest.approx(0.70)
    assert latest_score("FPT", "2026-11-23", db_path=db)["date"] == "2026-11-20"


def test_no_score_when_model_was_trained_on_the_analysis_date(db):
    # A backtest day inside the training window must not see this model's scores.
    assert latest_score("FPT", "2026-03-10", db_path=db) is None


def test_stale_scores_are_ignored(db):
    assert latest_score("FPT", "2026-12-31", db_path=db) is None


def test_missing_table_or_database(tmp_path):
    assert latest_score("FPT", "2026-11-20", db_path=tmp_path / "none.db") is None
    empty = tmp_path / "empty.db"
    sqlite3.connect(empty).close()
    assert latest_score("FPT", "2026-11-20", db_path=empty) is None


def test_quant_prompt_line():
    from ta_core.agents.quant_agent import _qlib_line

    assert "chưa có điểm" in _qlib_line({"qlib_rank_pct": None})
    line = _qlib_line({"qlib_rank_pct": 0.9, "qlib_score_date": "2026-11-20", "qlib_test_rank_ic": 0.031})
    assert "90/100" in line and "0.031" in line
