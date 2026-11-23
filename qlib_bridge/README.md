# Qlib bridge — factor research for Vietnamese stocks

Adds Microsoft Qlib's **Alpha158** factors and a **LightGBM** model to TradingAgent-VN.
The quant agent then sees, for each stock, where the model ranks it against the rest
of the basket, and how well the model predicted on data it was not trained on.

Qlib runs in **its own Python environment**. Its dependencies (MLflow, gym, numpy pins)
clash with the main app's, so the two only share files:

```
data/vnstock.db ──► qlib_bridge/run_qlib.py (Qlib environment)
                       1. export prices to CSV
                       2. convert with Qlib's scripts/dump_bin.py (from your Qlib checkout)
                       3. Alpha158 factors + LightGBM
                ◄── table qlib_scores in data/vnstock.db
ta_core/tools/qlib_scores.py reads it → quant agent prompt
```

Nothing from Qlib is copied into this repository; `dump_bin.py` is used from your
Qlib checkout.

## Setup (Windows 11, PowerShell)

Qlib supports Python 3.8–3.12. Python 3.11 is recommended.

```powershell
# From the TradingAgent-VN folder. Your Qlib checkout is expected at ..\qlib
py -3.11 -m venv .venv-qlib
.\.venv-qlib\Scripts\python.exe -m pip install -U pip
.\.venv-qlib\Scripts\python.exe -m pip install -r qlib_bridge\requirements-qlib.txt
```

Add to `.env`:

```env
QLIB_PYTHON=.venv-qlib\Scripts\python.exe
QLIB_REPO_DIR=..\qlib            # your Qlib checkout (default: ..\qlib)
# QLIB_WORKDIR=data/qlib         # where Qlib data and MLflow logs go
```

## Run

```powershell
python run.py crawl-vnstock --tickers VN30      # prices first: at least ~1 year per stock
python run.py qlib-refresh                      # export → Qlib → train → scores
```

Re-run `qlib-refresh` after new prices arrive (for example once a week). It takes about
a minute for VN30 on a normal PC.

## How the scores are made, and their limits

- **Split by time:** the first 60% of trading days train the model, the next 20%
  validate it (early stopping), and the last 20% are the test period. **Only test-period
  scores are stored.** They come from a model that never saw those days.
- **Out-of-sample Rank IC** (`test_rank_ic`) is the average daily rank correlation
  between the score and the next day's return over the test period. Around 0.02–0.05
  is a usable but weak signal; near or below 0 means no signal. The quant agent is told
  this number, and is told the score is only a reference when it is low.
- **No look-ahead in backtests:** the app uses a score only when the model was trained
  before the analysis date (`trained_until < ref_date`). Backtest days inside the
  training window get no Qlib score rather than one that has seen the future.
- **Scores older than 10 days are ignored**, so stale runs do not influence analysis.
- **Prices:** vn-market-data's prices are already adjusted for dividends and splits, so
  the adjustment factor is 1. The VWAP field is the typical price (high+low+close)/3,
  because the database stores no traded value per session.
- **Qlib does not simulate trading here.** Trading rules for Vietnam (100-share lots,
  T+2 settlement, fees, price bands) stay in TradingAgent-VN's own backtester.
- With VN30 only, the basket is small (about 30 stocks), so cross-sectional ranks are
  coarse. More stocks give a better model.
