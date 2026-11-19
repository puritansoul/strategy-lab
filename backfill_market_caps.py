#!/usr/bin/env python3
"""
backfill_market_caps.py  —  Fill 2001-2011 market cap gaps using FMP API.

Get a free key at: https://financialmodelingprep.com/developer/docs/
Free tier limit: 250 requests/day.  Run once per day until complete (~3 days).
Progress is saved to data/fmp_progress.json so each run resumes where it left off.

Usage:
    FMP_API_KEY=your_key_here python3 backfill_market_caps.py

    # Test with one ticker first:
    FMP_API_KEY=your_key python3 backfill_market_caps.py --test AAPL
"""

import argparse
import json
import os
import time
from pathlib import Path

import pandas as pd
import requests

# ── Config ────────────────────────────────────────────────────────────────────
DATA_DIR        = Path("data")
PARQUET         = DATA_DIR / "market_caps.parquet"
PROGRESS_FILE   = DATA_DIR / "fmp_progress.json"
FILL_YEARS      = list(range(2001, 2012))   # 2001–2011 inclusive
REQUESTS_PER_RUN = 240                       # stay under 250/day
SLEEP_SEC       = 0.5                        # between requests
FMP_BASE        = "https://financialmodelingprep.com/api/v3"

# ── FMP fetch ─────────────────────────────────────────────────────────────────

def fmp_market_caps(symbol: str, api_key: str) -> dict:
    """Return {year: market_cap_at_jan1} for FILL_YEARS from FMP."""
    url = (
        f"{FMP_BASE}/historical-market-capitalization/{symbol}"
        f"?from=2001-01-01&to=2012-06-01&limit=4000&apikey={api_key}"
    )
    try:
        r = requests.get(url, timeout=20)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        print(f"  ⚠  request error: {e}")
        return {}

    if not data or isinstance(data, dict):   # error response e.g. {"Error Message": ...}
        msg = data.get("Error Message", "") if isinstance(data, dict) else ""
        if msg:
            print(f"  ⚠  FMP error: {msg}")
        return {}

    df = pd.DataFrame(data)
    if df.empty or "date" not in df.columns or "marketCap" not in df.columns:
        return {}

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    result = {}
    for year in FILL_YEARS:
        jan1 = pd.Timestamp(f"{year}-01-01")
        # First trading day on or after Jan 1 within a 10-day window
        window = df[(df["date"] >= jan1) & (df["date"] <= jan1 + pd.DateOffset(days=10))]
        if not window.empty:
            result[year] = float(window.iloc[0]["marketCap"])
    return result

# ── Progress helpers ──────────────────────────────────────────────────────────

def load_progress() -> dict:
    if PROGRESS_FILE.exists():
        return json.loads(PROGRESS_FILE.read_text())
    return {"done": [], "data": {}}


def save_progress(p: dict) -> None:
    PROGRESS_FILE.write_text(json.dumps(p, indent=2))

# ── Merge + save parquet ──────────────────────────────────────────────────────

def merge_and_save(progress: dict) -> None:
    if not PARQUET.exists():
        print("⚠  market_caps.parquet not found — nothing to merge into.")
        return

    mc = pd.read_parquet(PARQUET)

    # Build FMP dataframe: index=year, columns=tickers
    fmp_rows: dict[int, dict] = {}
    for ticker, year_caps in progress["data"].items():
        for yr_str, cap in year_caps.items():
            yr = int(yr_str)
            fmp_rows.setdefault(yr, {})[ticker] = cap

    if not fmp_rows:
        print("No FMP data collected yet — nothing to merge.")
        return

    fmp_df = pd.DataFrame(fmp_rows).T.sort_index()
    fmp_df.index = fmp_df.index.astype(int)

    # For each FMP year/ticker, fill NaN cells in existing parquet (never overwrite existing data)
    for year in fmp_df.index:
        if year not in mc.index:
            mc.loc[year] = pd.NA
        for ticker in fmp_df.columns:
            val = fmp_df.loc[year, ticker]
            if pd.isna(val):
                continue
            if ticker not in mc.columns:
                mc[ticker] = pd.NA
            if pd.isna(mc.loc[year, ticker]):
                mc.loc[year, ticker] = val

    mc = mc.sort_index()
    mc.to_parquet(PARQUET)
    print(f"✅ Saved updated parquet ({PARQUET})")

    # Print coverage improvement
    coverage = mc.notna().sum(axis=1)
    print("\nCoverage after merge:")
    for y in sorted(mc.index):
        if y <= 2014:
            print(f"  {y}: {int(coverage[y])} tickers")

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", metavar="TICKER", help="Test with a single ticker and exit")
    args = parser.parse_args()

    api_key = os.environ.get("FMP_API_KEY", "").strip()
    if not api_key:
        print("❌  Set FMP_API_KEY environment variable.")
        print("    Get a free key at https://financialmodelingprep.com/developer/docs/")
        return

    # ── Test mode ────────────────────────────────────────────────────────────
    if args.test:
        tk = args.test.upper()
        print(f"Testing with {tk}...")
        caps = fmp_market_caps(tk, api_key)
        if not caps:
            print("No data returned. Check your API key and whether FMP covers this ticker.")
        else:
            print(f"Years found: {sorted(caps.keys())}")
            for yr, cap in sorted(caps.items()):
                print(f"  {yr}: ${cap/1e9:.1f}B")
        return

    # ── Collect tickers ───────────────────────────────────────────────────────
    tickers: set = set()
    comp_csv = DATA_DIR / "sp500_composition.csv"
    if comp_csv.exists():
        comp = pd.read_csv(comp_csv)
        # tickers column might be comma-separated strings per row
        for val in comp.iloc[:, 1].dropna():
            tickers.update(str(val).split(","))

    if PARQUET.exists():
        mc_cols = pd.read_parquet(PARQUET).columns.tolist()
        tickers.update(mc_cols)

    # Clean up: drop blank/junk symbols
    tickers = sorted(t.strip() for t in tickers if t.strip() and "." not in t and len(t) <= 5)
    print(f"Total unique tickers: {len(tickers)}")

    # ── Load progress ─────────────────────────────────────────────────────────
    progress = load_progress()
    remaining = [t for t in tickers if t not in progress["done"]]
    pct = 100 * len(progress["done"]) / len(tickers) if tickers else 0
    print(f"Done: {len(progress['done'])} ({pct:.0f}%)  |  Remaining: {len(remaining)}")
    print(f"This run will fetch up to {REQUESTS_PER_RUN} tickers.\n")

    if not remaining:
        print("All tickers already fetched — merging into parquet.")
        merge_and_save(progress)
        return

    # ── Fetch loop ────────────────────────────────────────────────────────────
    fetched = 0
    for ticker in remaining:
        if fetched >= REQUESTS_PER_RUN:
            print(f"\n⏸  Reached {REQUESTS_PER_RUN}-request limit for today.")
            print("   Run the script again tomorrow to continue.")
            break

        label = f"[{len(progress['done'])+1}/{len(tickers)}]"
        print(f"  {label} {ticker} ...", end=" ", flush=True)
        caps = fmp_market_caps(ticker, api_key)
        yrs  = sorted(caps.keys())
        if yrs:
            print(f"{len(yrs)} years ({yrs[0]}–{yrs[-1]})")
        else:
            print("no historical data")

        progress["done"].append(ticker)
        if caps:
            progress["data"][ticker] = {str(y): v for y, v in caps.items()}

        fetched += 1
        save_progress(progress)
        time.sleep(SLEEP_SEC)

    # ── Always merge what we have so far ─────────────────────────────────────
    print(f"\nMerging {len(progress['data'])} tickers into parquet...")
    merge_and_save(progress)

    if remaining[fetched:]:
        days_left = -(-len(remaining[fetched:]) // REQUESTS_PER_RUN)   # ceiling div
        print(f"\n~{days_left} more day(s) of runs needed to complete all tickers.")
    else:
        print("\n🎉 All tickers fetched and merged!")
        PROGRESS_FILE.unlink(missing_ok=True)
        print("Progress file cleaned up.")


if __name__ == "__main__":
    main()
