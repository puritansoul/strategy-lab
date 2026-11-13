"""
Supabase persistence for saved backtests.
Gracefully no-ops if credentials are missing — app still works without persistence.
"""

import copy
import json
import os
from typing import Optional

import pandas as pd
import requests


def _config():
    try:
        import streamlit as st
        url = os.environ.get("SUPABASE_URL") or st.secrets.get("SUPABASE_URL", "")
        key = os.environ.get("SUPABASE_KEY") or st.secrets.get("SUPABASE_KEY", "")
    except Exception:
        url = os.environ.get("SUPABASE_URL", "")
        key = os.environ.get("SUPABASE_KEY", "")
    return url.rstrip("/"), key


def _headers(key: str) -> dict:
    return {
        "apikey": key,
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        "Prefer": "return=representation",
    }


def _serialize(backtest: dict) -> dict:
    """Convert Timestamps → ISO strings so the dict is JSON-serializable."""
    b = copy.deepcopy(backtest)
    b.pop("_id", None)  # strip internal ID before storing in data column
    for row in b.get("strategy_rows", []):
        for field in ("buy_date", "sell_date"):
            v = row.get(field)
            if v is not None and hasattr(v, "isoformat"):
                row[field] = v.isoformat()
    return b


def _deserialize(data: dict) -> dict:
    """Convert ISO strings back to Timestamps."""
    for row in data.get("strategy_rows", []):
        for field in ("buy_date", "sell_date"):
            v = row.get(field)
            if v and isinstance(v, str):
                try:
                    row[field] = pd.Timestamp(v)
                except Exception:
                    row[field] = None
    return data


def load_backtests() -> list:
    url, key = _config()
    if not url or not key:
        return []
    try:
        r = requests.get(
            f"{url}/rest/v1/saved_backtests?select=*&order=created_at.asc",
            headers=_headers(key),
            timeout=10,
        )
        if not r.ok:
            return []
        return [
            {"_id": row["id"], **_deserialize(row["data"])}
            for row in r.json()
        ]
    except Exception:
        return []


def save_backtest(backtest: dict) -> Optional[str]:
    """Persist a backtest and return its UUID, or None on failure."""
    url, key = _config()
    if not url or not key:
        return None
    try:
        r = requests.post(
            f"{url}/rest/v1/saved_backtests",
            headers=_headers(key),
            json={"name": backtest.get("name", ""), "data": _serialize(backtest)},
            timeout=10,
        )
        if r.ok:
            rows = r.json()
            if rows:
                return rows[0]["id"]
    except Exception:
        pass
    return None


def delete_backtest(backtest_id: str) -> None:
    url, key = _config()
    if not url or not key:
        return
    try:
        requests.delete(
            f"{url}/rest/v1/saved_backtests?id=eq.{backtest_id}",
            headers=_headers(key),
            timeout=10,
        )
    except Exception:
        pass


def delete_all_backtests() -> None:
    url, key = _config()
    if not url or not key:
        return
    try:
        requests.delete(
            f"{url}/rest/v1/saved_backtests?id=neq.00000000-0000-0000-0000-000000000000",
            headers=_headers(key),
            timeout=10,
        )
    except Exception:
        pass


def load_search_history() -> list:
    url, key = _config()
    if not url or not key:
        return []
    try:
        r = requests.get(
            f"{url}/rest/v1/search_history?select=searches&id=eq.1",
            headers=_headers(key),
            timeout=10,
        )
        if r.ok and r.json():
            return r.json()[0].get("searches", [])
    except Exception:
        pass
    return []


def save_search_history(searches: list) -> None:
    url, key = _config()
    if not url or not key:
        return
    try:
        requests.patch(
            f"{url}/rest/v1/search_history?id=eq.1",
            headers={**_headers(key), "Prefer": "return=minimal"},
            json={"searches": searches[:15]},
            timeout=10,
        )
    except Exception:
        pass
