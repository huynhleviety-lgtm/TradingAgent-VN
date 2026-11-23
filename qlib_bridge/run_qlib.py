"""Qlib factor research for Vietnamese stocks: Alpha158 factors + LightGBM scores.

Runs inside the separate Qlib Python environment (see qlib_bridge/README.md), never
inside the main app's environment: Qlib pins packages (mlflow, gym, numpy, ...) that
clash with TradingAgent-VN's. The two sides share only files:

    data/vnstock.db --(this script)--> Qlib binary data --> Alpha158 + LightGBM
                    <-- table qlib_scores (ticker, date, score, rank_pct, ...)

The main app reads ``qlib_scores`` (ta_core/tools/qlib_scores.py).

Conversion to Qlib's format uses Qlib's own ``scripts/dump_bin.py`` from your Qlib
checkout (``--qlib-repo``); it is not copied here. The script uses only the standard
library, pandas and Qlib, so it does not import any TradingAgent-VN module.

Usage (from the TradingAgent-VN folder, with the Qlib environment's Python):
    <qlib-python> qlib_bridge/run_qlib.py --db data/vnstock.db --qlib-repo ../qlib
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

INDEX_SYMBOLS = {"VNINDEX", "VN30", "HNXINDEX", "HNX30", "UPCOMINDEX", "VN100"}
FIELDS = ["open", "high", "low", "close", "volume", "vwap", "factor"]
MODEL_NAME = "alpha158_lgbm"


def log(message: str) -> None:
    print(f"[qlib] {message}", flush=True)


# ── 1. Export prices from data/vnstock.db to one CSV per symbol ─────────────────

def export_csv(db_path: Path, csv_dir: Path, min_rows: int) -> list[str]:
    with sqlite3.connect(db_path) as con:
        df = pd.read_sql_query(
            "SELECT ticker, date, open, high, low, close, volume FROM market_data_daily "
            "WHERE close > 0 ORDER BY ticker, date",
            con,
            parse_dates=["date"],
        )
    if df.empty:
        raise SystemExit(f"No prices in {db_path}. Run: python run.py crawl-vnstock --tickers VN30")

    if csv_dir.exists():
        shutil.rmtree(csv_dir)
    csv_dir.mkdir(parents=True)

    df["date"] = df["date"].dt.normalize()
    df = df.drop_duplicates(["ticker", "date"], keep="last")
    # Typical price as a VWAP stand-in: the DB stores no traded value per session.
    df["vwap"] = (df["high"] + df["low"] + df["close"]) / 3.0
    # Prices from vn-market-data are already adjusted for corporate actions.
    df["factor"] = 1.0

    written = []
    for ticker, rows in df.groupby("ticker"):
        if ticker not in INDEX_SYMBOLS and len(rows) < min_rows:
            log(f"skip {ticker}: only {len(rows)} sessions (need {min_rows})")
            continue
        out = rows.drop(columns=["ticker"]).copy()
        out["symbol"] = ticker
        out["date"] = out["date"].dt.strftime("%Y-%m-%d")
        out.to_csv(csv_dir / f"{ticker}.csv", index=False)
        written.append(ticker)
    log(f"exported {len(written)} symbols to {csv_dir}")
    return written


# ── 2. Convert CSVs to Qlib's binary format with Qlib's own dump_bin.py ─────────

def dump_to_qlib(qlib_repo: Path, csv_dir: Path, qlib_dir: Path) -> None:
    script = qlib_repo / "scripts" / "dump_bin.py"
    if not script.is_file():
        raise SystemExit(f"Qlib's dump_bin.py not found at {script}. Pass --qlib-repo <your qlib checkout>.")
    # dump_bin starts worker processes that re-import it by name, so it must be a normal
    # import from its own folder (child processes inherit sys.path, also on Windows).
    scripts_dir = str(script.parent)
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    module = importlib.import_module("dump_bin")

    if qlib_dir.exists():
        shutil.rmtree(qlib_dir)
    module.DumpDataAll(
        data_path=str(csv_dir),
        qlib_dir=str(qlib_dir),
        freq="day",
        max_workers=1,  # one process: avoids Windows spawn overhead and pickling issues
        date_field_name="date",
        symbol_field_name="symbol",
        include_fields=",".join(FIELDS),
    ).dump()
    log(f"Qlib data written to {qlib_dir}")


# ── 3. Alpha158 factors + LightGBM, then score every stock ──────────────────────

def split_segments(calendar: list[pd.Timestamp], train_frac: float, valid_frac: float):
    n = len(calendar)
    train_end = int(n * train_frac)
    valid_end = int(n * (train_frac + valid_frac))
    return {
        "train": (calendar[0], calendar[train_end - 1]),
        "valid": (calendar[train_end], calendar[valid_end - 1]),
        "test": (calendar[valid_end], calendar[-1]),
    }


def train_and_score(qlib_dir: Path, stocks: list[str], args) -> tuple[pd.DataFrame, dict]:
    import qlib
    from qlib.contrib.data.handler import Alpha158
    from qlib.contrib.model.gbdt import LGBModel
    from qlib.data import D
    from qlib.data.dataset import DatasetH

    # region "cn" only sets defaults such as the trade unit; this script does not
    # simulate trading, so Vietnam-specific rules do not apply here.
    # Qlib logs each model fit to MLflow. Keep that log next to the Qlib data rather
    # than in whatever folder the script was started from.
    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    mlruns = (qlib_dir.parent / "mlruns").resolve()
    qlib.init(
        provider_uri=str(qlib_dir),
        region="cn",
        kernels=1,
        exp_manager={
            "class": "MLflowExpManager",
            "module_path": "qlib.workflow.expm",
            "kwargs": {"uri": mlruns.as_uri(), "default_exp_name": "tradingagent_vn"},
        },
    )

    calendar = list(D.calendar(freq="day"))
    if len(calendar) < args.min_sessions:
        raise SystemExit(f"Only {len(calendar)} sessions of history; need at least {args.min_sessions}.")
    seg = split_segments(calendar, args.train_frac, args.valid_frac)
    log("segments: " + ", ".join(f"{k}={a.date()}..{b.date()}" for k, (a, b) in seg.items()))

    handler = Alpha158(
        instruments=stocks,
        start_time=calendar[0],
        end_time=calendar[-1],
        fit_start_time=seg["train"][0],
        fit_end_time=seg["train"][1],
    )
    dataset = DatasetH(handler=handler, segments=seg)

    model = LGBModel(
        loss="mse",
        learning_rate=0.05,
        num_leaves=31,
        max_depth=6,
        num_boost_round=args.rounds,
        early_stopping_rounds=50,
        num_threads=args.threads,
        verbosity=-1,
    )
    model.fit(dataset)

    test_pred = model.predict(dataset, segment="test")
    label = dataset.prepare("test", col_set="label").iloc[:, 0]
    joined = pd.concat([test_pred.rename("score"), label.rename("label")], axis=1).dropna()
    daily_ic = joined.groupby(level="datetime").apply(
        lambda g: g["score"].corr(g["label"], method="spearman") if len(g) > 2 else np.nan
    ).dropna()
    metrics = {
        "rank_ic_mean": float(daily_ic.mean()) if len(daily_ic) else None,
        "rank_ic_ir": float(daily_ic.mean() / daily_ic.std()) if len(daily_ic) > 1 and daily_ic.std() else None,
        "test_days": int(len(daily_ic)),
    }
    log(f"out-of-sample Rank IC mean={metrics['rank_ic_mean']} IR={metrics['rank_ic_ir']} over {metrics['test_days']} days")

    # Only test-period scores are stored: they come from a model that never saw those dates.
    scores = test_pred.rename("score").reset_index()
    scores.columns = ["date", "ticker", "score"]
    scores["rank_pct"] = scores.groupby("date")["score"].rank(pct=True)
    meta = {
        "train_start": str(seg["train"][0].date()),
        "train_end": str(seg["train"][1].date()),
        "valid_end": str(seg["valid"][1].date()),
        **metrics,
    }
    return scores, meta


# ── 4. Write scores back to data/vnstock.db ─────────────────────────────────────

def write_scores(db_path: Path, scores: pd.DataFrame, meta: dict) -> None:
    run_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    rows = [
        (
            r.ticker,
            pd.Timestamp(r.date).strftime("%Y-%m-%d"),
            float(r.score),
            float(r.rank_pct),
            MODEL_NAME,
            meta["valid_end"],
            meta["rank_ic_mean"],
            run_at,
        )
        for r in scores.itertuples(index=False)
    ]
    with sqlite3.connect(db_path) as con:
        con.execute(
            """CREATE TABLE IF NOT EXISTS qlib_scores (
                   ticker TEXT NOT NULL,
                   date TEXT NOT NULL,
                   score REAL,
                   rank_pct REAL,
                   model TEXT NOT NULL,
                   trained_until TEXT NOT NULL,
                   test_rank_ic REAL,
                   run_at TEXT NOT NULL,
                   PRIMARY KEY (ticker, date, model)
               )"""
        )
        con.execute("DELETE FROM qlib_scores WHERE model = ?", (MODEL_NAME,))
        con.executemany("INSERT INTO qlib_scores VALUES (?,?,?,?,?,?,?,?)", rows)
        con.execute(
            "CREATE TABLE IF NOT EXISTS qlib_runs (run_at TEXT PRIMARY KEY, model TEXT, meta TEXT)"
        )
        con.execute("INSERT INTO qlib_runs VALUES (?,?,?)", (run_at, MODEL_NAME, json.dumps(meta)))
    log(f"wrote {len(rows)} scores to {db_path} (table qlib_scores)")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--db", type=Path, required=True, help="Path to data/vnstock.db")
    parser.add_argument("--qlib-repo", type=Path, required=True, help="Your Qlib checkout (for scripts/dump_bin.py)")
    parser.add_argument("--workdir", type=Path, default=Path("data/qlib"), help="Where Qlib data is written")
    parser.add_argument("--train-frac", type=float, default=0.6)
    parser.add_argument("--valid-frac", type=float, default=0.2)
    parser.add_argument("--min-sessions", type=int, default=250)
    parser.add_argument("--min-rows", type=int, default=120, help="Skip stocks with less history")
    parser.add_argument("--rounds", type=int, default=500)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args(argv)

    db_path = args.db.resolve()
    if not db_path.is_file():
        raise SystemExit(f"Database not found: {db_path}")
    workdir = args.workdir.resolve()

    symbols = export_csv(db_path, workdir / "csv", args.min_rows)
    stocks = [s for s in symbols if s not in INDEX_SYMBOLS]
    if len(stocks) < 5:
        raise SystemExit(f"Need at least 5 stocks for cross-sectional factors; have {len(stocks)}.")
    dump_to_qlib(args.qlib_repo.resolve(), workdir / "csv", workdir / "qlib_data")
    scores, meta = train_and_score(workdir / "qlib_data", stocks, args)
    write_scores(db_path, scores, meta)
    log("done: " + json.dumps(meta))
    return 0


if __name__ == "__main__":  # required on Windows: Qlib/joblib start worker processes
    sys.exit(main())
