"""System status for the dashboard: local AI (Ollama), market data and Qlib.

Every check is read-only and quick, and never raises: a failing component is
reported as not OK with a short reason, so the status page always renders.
"""
from __future__ import annotations

import asyncio
import json
import sqlite3
import time
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx

from config import models, paths, qlib_settings

INDEX_SYMBOLS = {"VNINDEX", "VN30", "HNXINDEX", "HNX30", "UPCOMINDEX", "VN100"}
STALE_PRICE_DAYS = 4  # calendar days; covers a weekend plus one holiday


def _ro(db: Path) -> sqlite3.Connection:
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    return con


# ── Local AI ────────────────────────────────────────────────────────────────

def _configured_models() -> Dict[str, str]:
    return {
        "analysts": models.t2_technical_model,
        "debate": models.t3_debate_model,
        "cio": models.t4_cio_model,
        "news": models.news_model,
    }


async def llm_status() -> Dict[str, Any]:
    base = models.llm_base_url
    configured = _configured_models()
    out: Dict[str, Any] = {
        "base_url": base,
        "configured": configured,
        "concurrency": models.llm_concurrency,
        "reachable": False,
        "available": [],
        "missing": [],
        "ok": False,
        "message": "",
    }
    # Ollama lists pulled models at /api/tags; other OpenAI-compatible servers at /v1/models.
    ollama_root = base[:-3] if base.endswith("/v1") else base
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            try:
                resp = await client.get(f"{ollama_root}/api/tags")
                resp.raise_for_status()
                available = [m.get("name", "") for m in resp.json().get("models", [])]
                out["server"] = "ollama"
            except (httpx.HTTPError, ValueError):
                resp = await client.get(f"{base}/models", headers={"Authorization": f"Bearer {models.llm_api_key}"})
                resp.raise_for_status()
                available = [m.get("id", "") for m in resp.json().get("data", [])]
                out["server"] = "openai-compatible"
    except Exception as exc:  # pylint: disable=broad-except
        out["message"] = f"Không kết nối được {base}: {type(exc).__name__}. Ollama đã chạy chưa?"
        return out

    out["reachable"] = True
    out["available"] = sorted(a for a in available if a)
    names = set(out["available"]) | {a.split(":")[0] for a in out["available"] if a.endswith(":latest")}
    out["missing"] = sorted({m for m in configured.values() if m and m not in names})
    out["ok"] = not out["missing"]
    out["message"] = (
        "Sẵn sàng."
        if out["ok"]
        else "Thiếu model: " + ", ".join(out["missing"]) + " — chạy: ollama pull " + " ".join(out["missing"])
    )
    return out


# ── Market data ─────────────────────────────────────────────────────────────

def market_data_status() -> Dict[str, Any]:
    db = Path(paths.vnstock_db_path)
    out: Dict[str, Any] = {
        "db_path": str(db),
        "cache_path": str(paths.market_data_cache_path),
        "cache_exists": Path(paths.market_data_cache_path).is_file(),
        "ok": False,
        "stocks": 0,
        "indices": [],
        "latest_date": None,
        "stale_days": None,
        "stale_tickers": [],
        "ratio_tickers": 0,
        "message": "",
    }
    if not db.is_file():
        out["message"] = "Chưa có data/vnstock.db — chạy: python run.py crawl-vnstock --tickers VN30"
        return out
    try:
        with _ro(db) as con:
            rows = con.execute(
                "SELECT ticker, MAX(date) AS last FROM market_data_daily GROUP BY ticker"
            ).fetchall()
            try:
                out["ratio_tickers"] = con.execute(
                    "SELECT COUNT(DISTINCT ticker) FROM financial_ratios"
                ).fetchone()[0]
            except sqlite3.OperationalError:
                pass
    except sqlite3.Error as exc:
        out["message"] = f"Không đọc được database: {exc}"
        return out
    if not rows:
        out["message"] = "Database chưa có giá — chạy: python run.py crawl-vnstock --tickers VN30"
        return out

    today = date.today()
    last_by_ticker = {r["ticker"]: str(r["last"])[:10] for r in rows}
    out["stocks"] = sum(1 for t in last_by_ticker if t not in INDEX_SYMBOLS)
    out["indices"] = sorted(t for t in last_by_ticker if t in INDEX_SYMBOLS)
    latest = max(last_by_ticker.values())
    out["latest_date"] = latest
    out["stale_days"] = (today - date.fromisoformat(latest)).days
    out["stale_tickers"] = sorted(
        t for t, d in last_by_ticker.items() if (today - date.fromisoformat(d)).days > STALE_PRICE_DAYS
    )
    out["ok"] = out["stale_days"] <= STALE_PRICE_DAYS and not out["stale_tickers"]
    out["message"] = (
        "Giá đã cập nhật."
        if out["ok"]
        else f"Có {len(out['stale_tickers'])} mã chưa cập nhật quá {STALE_PRICE_DAYS} ngày — bấm Đồng bộ hoặc chạy crawl-vnstock."
    )
    return out


# ── Qlib ────────────────────────────────────────────────────────────────────

def _latest_run(db: Path) -> Optional[Dict[str, Any]]:
    try:
        with _ro(db) as con:
            row = con.execute("SELECT run_at, model, meta FROM qlib_runs ORDER BY run_at DESC LIMIT 1").fetchone()
    except sqlite3.Error:
        return None
    if row is None:
        return None
    meta = json.loads(row["meta"] or "{}")
    return {"run_at": row["run_at"], "model": row["model"], **meta}


def qlib_scores(limit: int = 100) -> Dict[str, Any]:
    """Latest-day cross-sectional ranking, best first."""
    db = Path(paths.vnstock_db_path)
    out: Dict[str, Any] = {"date": None, "rows": [], "run": None}
    if not db.is_file():
        return out
    out["run"] = _latest_run(db)
    try:
        with _ro(db) as con:
            latest = con.execute("SELECT MAX(date) FROM qlib_scores").fetchone()[0]
            if latest is None:
                return out
            rows = con.execute(
                "SELECT ticker, score, rank_pct FROM qlib_scores WHERE date = ? ORDER BY score DESC LIMIT ?",
                (latest, limit),
            ).fetchall()
    except sqlite3.Error:
        return out
    out["date"] = latest
    out["rows"] = [dict(r) for r in rows]
    return out


_refresh_state: Dict[str, Any] = {"running": False, "started_at": None, "finished_at": None, "exit_code": None}
_refresh_lock = asyncio.Lock()


def qlib_status() -> Dict[str, Any]:
    db = Path(paths.vnstock_db_path)
    python = qlib_settings.python
    repo = Path(qlib_settings.repo_dir)
    run = _latest_run(db) if db.is_file() else None
    out: Dict[str, Any] = {
        "python": python,
        "python_ok": bool(python) and Path(python).exists(),
        "repo_dir": str(repo),
        "repo_ok": (repo / "scripts" / "dump_bin.py").is_file(),
        "last_run": run,
        "refresh": dict(_refresh_state),
        "ok": False,
        "message": "",
    }
    if not out["python_ok"]:
        out["message"] = "Chưa cấu hình QLIB_PYTHON (môi trường Qlib riêng) — xem qlib_bridge/README.md."
    elif not out["repo_ok"]:
        out["message"] = f"Không thấy scripts/dump_bin.py trong {repo} — đặt QLIB_REPO_DIR."
    elif run is None:
        out["message"] = "Qlib chưa chạy lần nào — bấm Chạy Qlib."
    else:
        age = (datetime.now().astimezone() - datetime.fromisoformat(run["run_at"])).days
        ic = run.get("rank_ic_mean")
        out["ok"] = age <= 10
        weak = " Tín hiệu yếu (IC thấp), chỉ tham khảo." if ic is None or ic < 0.02 else ""
        out["message"] = (f"Chạy {age} ngày trước." if out["ok"] else f"Kết quả đã cũ ({age} ngày) — chạy lại.") + weak
    return out


async def start_qlib_refresh() -> Dict[str, Any]:
    """Start a Qlib run in the background; only one at a time."""
    if _refresh_lock.locked():
        return {"started": False, "reason": "Qlib đang chạy.", **_refresh_state}

    async def _run() -> None:
        async with _refresh_lock:
            from ta_core.tools.qlib_scores import refresh

            _refresh_state.update(running=True, started_at=time.time(), finished_at=None, exit_code=None)
            try:
                code = await asyncio.to_thread(refresh)
            except Exception:  # pylint: disable=broad-except  (e.g. QLIB_PYTHON missing)
                code = -1
            _refresh_state.update(running=False, finished_at=time.time(), exit_code=code)

    asyncio.create_task(_run())
    await asyncio.sleep(0)  # let the task take the lock before replying
    return {"started": True, **_refresh_state}


async def full_status() -> Dict[str, Any]:
    llm = await llm_status()
    return {
        "llm": llm,
        "market_data": await asyncio.to_thread(market_data_status),
        "qlib": await asyncio.to_thread(qlib_status),
        "checked_at": datetime.now().astimezone().isoformat(timespec="seconds"),
    }
