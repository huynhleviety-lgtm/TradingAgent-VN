from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

# Load .env at project root
PROJECT_ROOT = Path(__file__).resolve().parent
load_dotenv(dotenv_path=PROJECT_ROOT / ".env")


def _resolve_project_path(raw: str) -> Path:
    path = Path(raw)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _env_pct(name: str, default: float) -> float:
    raw = float(os.getenv(name, str(default)))
    return raw * 100.0 if 0.0 < raw <= 1.0 else raw


@dataclass(frozen=True)
class TradingConfig:
    portfolio_cash: float = float(os.getenv("PORTFOLIO_CASH", 1_000_000_000))
    lot_size: int = int(os.getenv("LOT_SIZE", 100))
    settlement_lag_days: int = int(os.getenv("SETTLEMENT_LAG_DAYS", 2))
    # HOSE fees: buy 0.15%, sell 0.25% (0.15% commission + 0.1% tax)
    buy_fee_rate: float = float(os.getenv("BUY_FEE_RATE", 0.0015))
    sell_fee_rate: float = float(os.getenv("SELL_FEE_RATE", 0.0025))
    max_trade_pct: float = _env_pct("MAX_TRADE_PCT", 25.0)
    max_buys_per_ticker: int = int(os.getenv("MAX_BUYS_PER_TICKER", 100))


@dataclass(frozen=True)
class StrategyThresholds:
    price_change_threshold: float = float(os.getenv("PRICE_CHANGE_THRESHOLD", 5.0))
    vol_ratio_threshold: float = float(os.getenv("VOL_RATIO_THRESHOLD", 1.5))
    news_min_count: int = int(os.getenv("NEWS_MIN_COUNT", 3))
    alpha_threshold: float = float(os.getenv("ALPHA_THRESHOLD", 60.0))
    sell_threshold_offset: float = float(os.getenv("SELL_THRESHOLD_OFFSET", 5.0))
    atr_scale: float = float(os.getenv("ATR_SCALE", 1.5))
    weight_alpha: float = float(os.getenv("WEIGHT_ALPHA", 0.6))
    weight_beta: float = float(os.getenv("WEIGHT_BETA", 0.4))
    weight_increment_buffer_pct: float = float(os.getenv("WEIGHT_INCREMENT_BUFFER_PCT", 0.5))
    news_lookback_days: int = int(os.getenv("NEWS_LOOKBACK_DAYS", 3))


@dataclass(frozen=True)
class RiskLimits:
    """Centralised risk guardrails for backtests and CIO decisions."""
    max_position_pct: float = _env_pct("MAX_POSITION_PCT", 15.0)
    max_portfolio_invested_pct: float = _env_pct("MAX_PORTFOLIO_INVESTED_PCT", 85.0)
    stop_loss_pct: float = _env_pct("STOP_LOSS_PCT", 7.0)
    max_drawdown_pct: float = _env_pct("MAX_DRAWDOWN_PCT", 15.0)
    max_sector_exposure_pct: float = _env_pct("MAX_SECTOR_EXPOSURE_PCT", 40.0)
    min_cash_reserve_pct: float = _env_pct("MIN_CASH_RESERVE_PCT", 15.0)


@dataclass(frozen=True)
class WorkflowWeights:
    trad_target_weight: float = float(os.getenv("TRAD_TARGET_WEIGHT", 10.0))
    kelly_min_weight_pct: float = float(os.getenv("KELLY_MIN_WEIGHT_PCT", 5.0))
    kelly_max_weight_pct: float = float(os.getenv("KELLY_MAX_WEIGHT_PCT", 40.0))


# ---------------------------------------------------------------------------
# Local AI (Ollama). Every agent talks to one OpenAI-compatible endpoint.
# Defaults point at Ollama on this machine; set LLM_BASE_URL / LLM_API_KEY to
# use any other OpenAI-compatible server. CLIPROXY_* are still read for
# backward compatibility when the LLM_* variables are not set.
# ---------------------------------------------------------------------------
OLLAMA_DEFAULT_BASE_URL = "http://127.0.0.1:11434/v1"
DEFAULT_LOCAL_MODEL = "qwen3:8b"


def _llm_base_url() -> str:
    raw = os.getenv("LLM_BASE_URL") or os.getenv("CLIPROXY_BASE_URL") or OLLAMA_DEFAULT_BASE_URL
    return raw.rstrip("/")


def _llm_api_key() -> str:
    # Ollama ignores the key but OpenAI-compatible clients require one.
    return os.getenv("LLM_API_KEY") or os.getenv("CLIPROXY_API_KEY") or "ollama"


def _model(*names: str, default: str | None = None) -> str:
    for name in names:
        value = os.getenv(name)
        if value:
            return value
    return default or os.getenv("LLM_MODEL") or DEFAULT_LOCAL_MODEL


@dataclass(frozen=True)
class ModelConfig:
    """Model per role. Each falls back to LLM_MODEL, so one variable sets them all.

    LLM_REASONING_MODEL (if set) is used for the debate and CIO roles, which
    benefit most from a larger model, e.g. LLM_MODEL=qwen3:8b and
    LLM_REASONING_MODEL=qwen3:14b.
    """

    primary_model: str = _model("PRIMARY_MODEL")
    financial_model: str = _model("FINANCIAL_MODEL")
    news_model: str = _model("NEWS_MODEL")

    # Tier 2: analysts
    t2_macro_model: str = _model("T2_MACRO", "T2_MACRO_PRIMARY")
    t2_news_model: str = _model("T2_NEWS", "T2_NEWS_PRIMARY")
    t2_financial_model: str = _model("T2_FINANCIAL", "T2_FINANCIAL_PRIMARY")
    t2_technical_model: str = _model("T2_TECHNICAL", "T2_TECHNICAL_PRIMARY")
    t2_quant_model: str = _model("T2_QUANT", "T2_QUANT_PRIMARY")

    # Tier 3: debate
    t3_debate_model: str = _model("T3_DEBATE", "T3_DEBATE_PRIMARY", "LLM_REASONING_MODEL")
    t3_argument_model: str = _model("T3_ARGUMENT", "T3_ARGUMENT_PRIMARY", "LLM_REASONING_MODEL")

    # Tier 4: CIO
    t4_cio_model: str = _model("T4_CIO", "T4_CIO_PRIMARY", "LLM_REASONING_MODEL")

    # Daily report
    daily_report_model: str = _model("DAILY_REPORT", "DAILY_REPORT_PRIMARY")

    # A local model serves a few requests at a time; match OLLAMA_NUM_PARALLEL.
    llm_concurrency: int = int(os.getenv("LLM_CONCURRENCY", 2))
    llm_base_url: str = field(default_factory=_llm_base_url)
    llm_api_key: str = field(default_factory=_llm_api_key)
    llm_timeout_seconds: float = float(os.getenv("LLM_TIMEOUT_SECONDS", 900))


@dataclass(frozen=True)
class PathConfig:
    data_dir: Path = field(default_factory=lambda: _resolve_project_path(os.getenv("DATA_DIR", "data")))
    vnstock_db_path: Path = field(
        default_factory=lambda: _resolve_project_path(os.getenv("VNSTOCK_DB_PATH", "data/vnstock.db"))
    )
    news_db_path: Path = field(
        default_factory=lambda: _resolve_project_path(os.getenv("NEWS_DB_PATH", "data/news.db"))
    )
    cognitive_db_path: Path = field(
        default_factory=lambda: _resolve_project_path(os.getenv("COGNITIVE_DB_PATH", "data/cognitive.db"))
    )
    market_data_cache_path: Path = field(
        default_factory=lambda: _resolve_project_path(os.getenv("VN_MARKET_DATA_DB", "data/vn_market_data.db"))
    )
    market_db_path: Path = field(
        default_factory=lambda: _resolve_project_path(os.getenv("MARKET_DB_PATH", "data/market.db"))
    )
    backtest_results_dir: Path = field(
        default_factory=lambda: _resolve_project_path(os.getenv("BACKTEST_RESULTS_DIR", "backtest_results"))
    )
    rag_storage_dir: Path = field(
        default_factory=lambda: _resolve_project_path(os.getenv("WORKDIR", "ta_core/rag_storage"))
    )
    analysis_reports_dir: Path = field(
        default_factory=lambda: _resolve_project_path(os.getenv("ANALYSIS_REPORTS_DIR", "ta_core/analysis_reports"))
    )


trading = TradingConfig()
strategy = StrategyThresholds()
risk_limits = RiskLimits()
workflow_weights = WorkflowWeights()
models = ModelConfig()
paths = PathConfig()
paths.data_dir.mkdir(parents=True, exist_ok=True)

__all__ = [
    "trading",
    "strategy",
    "risk_limits",
    "workflow_weights",
    "models",
    "paths",
    "PROJECT_ROOT",
]
