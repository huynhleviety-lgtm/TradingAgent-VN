"""System endpoints: component status and Qlib scores."""

from __future__ import annotations

from typing import Any, Dict

from fastapi import APIRouter, Query

from services import system_service

router = APIRouter()


@router.get("/system/status")
async def get_status() -> Dict[str, Any]:
    """Local AI (Ollama), market data freshness and Qlib, in one call."""
    return await system_service.full_status()


@router.get("/qlib/scores")
def get_qlib_scores(limit: int = Query(100, ge=1, le=500)) -> Dict[str, Any]:
    """Latest Qlib ranking (best first) and the run's out-of-sample metrics."""
    return system_service.qlib_scores(limit=limit)


@router.post("/qlib/refresh")
async def refresh_qlib() -> Dict[str, Any]:
    """Start a Qlib run in the background. Poll /system/status for progress."""
    return await system_service.start_qlib_refresh()
