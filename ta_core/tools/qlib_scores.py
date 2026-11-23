"""Read Qlib factor scores and launch the Qlib research run.

Qlib runs in its own Python environment (qlib_bridge/README.md), so this module never
imports Qlib. It reads the ``qlib_scores`` table that ``qlib_bridge/run_qlib.py``
writes into ``data/vnstock.db``, and can start that script with the Qlib Python.

Look-ahead guard: a score is only used for an analysis date when the model that
produced it was trained strictly before that date, and the score itself is dated on
or before it. Backtests on dates the model was trained on therefore get no score
rather than a score that has seen the future.
"""
from __future__ import annotations

import sqlite3
import subprocess
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

from config import PROJECT_ROOT, paths, qlib_settings

# A score older than this (calendar days) relative to the analysis date is ignored.
MAX_SCORE_AGE_DAYS = 10


def _as_date(value: str | date | datetime) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return datetime.strptime(str(value)[:10], "%Y-%m-%d").date()


def latest_score(
    ticker: str,
    ref_date: str | date | datetime,
    *,
    db_path: Path | None = None,
) -> dict[str, Any] | None:
    """Most recent usable Qlib score for ``ticker`` as of ``ref_date``, or None.

    Returns ``{date, score, rank_pct, model, trained_until, test_rank_ic}``. None when
    Qlib has not been run, the stock was not scored, or no score passes the
    look-ahead and staleness checks.
    """
    db = Path(db_path or paths.vnstock_db_path)
    if not db.is_file():
        return None
    ref = _as_date(ref_date)
    try:
        with sqlite3.connect(f"file:{db}?mode=ro", uri=True) as con:
            con.row_factory = sqlite3.Row
            row = con.execute(
                """SELECT date, score, rank_pct, model, trained_until, test_rank_ic
                   FROM qlib_scores
                   WHERE ticker = ? AND date <= ? AND trained_until < ?
                   ORDER BY date DESC LIMIT 1""",
                (ticker.strip().upper(), ref.isoformat(), ref.isoformat()),
            ).fetchone()
    except sqlite3.OperationalError:  # table not created yet: Qlib never ran
        return None
    if row is None:
        return None
    if (ref - _as_date(row["date"])) > timedelta(days=MAX_SCORE_AGE_DAYS):
        return None
    return dict(row)


def refresh(
    *,
    qlib_python: str | None = None,
    qlib_repo: Path | None = None,
    workdir: Path | None = None,
    extra_args: list[str] | None = None,
) -> int:
    """Run qlib_bridge/run_qlib.py with the Qlib environment's Python. Returns its exit code."""
    python = qlib_python or qlib_settings.python
    if not python:
        raise RuntimeError(
            "QLIB_PYTHON is not set. Point it at the Qlib environment's Python, e.g. "
            r"QLIB_PYTHON=.venv-qlib\Scripts\python.exe (see qlib_bridge/README.md)."
        )
    command = [
        python,
        str(PROJECT_ROOT / "qlib_bridge" / "run_qlib.py"),
        "--db", str(paths.vnstock_db_path),
        "--qlib-repo", str(qlib_repo or qlib_settings.repo_dir),
        "--workdir", str(workdir or qlib_settings.workdir),
        *(extra_args or []),
    ]
    return subprocess.run(command, cwd=PROJECT_ROOT, check=False).returncode
