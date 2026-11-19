#!/usr/bin/env python3
"""
backfill_market_caps.py  —  Fill 2001-2011 market cap gaps using yfinance.

No API key needed. Uses Yahoo Finance shares outstanding history × price.
Progress saved to data/yf_progress.json — resume anytime if interrupted.

Usage:
    python3 backfill_market_caps.py            # run (or resume) full backfill
    python3 backfill_market_caps.py --test AAPL  # test one ticker first
"""

import argparse
import json
import time
import warnings
from pathlib import Path

import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

# ── Config ────────────────────────────────────────────────────────────────────
DATA_DIR      = Path("data")
PARQUET       = DATA_DIR / "market_caps.parquet"
PROGRESS_FILE = DATA_DIR / "yf_progress.json"
PRICES_FILE   = DATA_DIR / "prices.parquet"
FILL_YEARS    = list(range(2001, 2012))   # 2001–2011 inclusive
SLEEP_SEC     = 0.3                       # polite delay between tickers

# ── Core fetch ────────────────────────────────────────────────────────────────

def yf_market_caps(symbol: str, prices_df: pd.DataFrame) -> dict:
    """
    Return {year: market_cap_at_jan1} for FILL_YEARS.
    Market cap = shares_outstanding × closing_price on first trading day of year.
    """
    try:
        tk = yf.Ticker(symbol)
        shares_series = tk.get_shares_full(start="2000-01-01", end="2012-06-01")
    except Exception as e:
        print(f"  ⚠  yfinance error: {e}")
        return {}

    if shares_series is None or shares_series.empty:
        return {}

    shares_series = shares_series.sort_index()

    result = {}
    for year in FILL_YEARS:
        jan1 = pd.Timestamp(f"{year}-01-01")
        jan10 = jan1 + pd.DateOffset(days=10)

        # Get price: first trading day on/after Jan 1 from prices.parquet
        price = None
        if symbol in prices_df.columns:
            price_slice = prices_df.loc[jan1:jan10, symbol].dropna()
            if not price_slice.empty:
                price = float(price_slice.iloc[0])

        # Fallback: fetch price directly from yfinance
        if price is None:
            try:
                hist = tk.history(start=jan1.date(), end=jan10.date(), auto_adjust=True)
                if not hist.empty:
                    price = float(hist["Close"].iloc[0])
            except Exception:
                pass

        if price is None:
            continue

        # Get shares: last known value on or before Jan 1
        past = shares_series[shares_series.index <= jan10]
        if past.empty:
            continue
        shares = float(past.iloc[-1])

        if shares >= 1_000_000:   # sanity: at least 1M shares
            result[year] = price * shares

    return result

# ── Progress helpers ──────────────────────────────────────────────────────────

def load_progress() -> dict:
    if PROGRESS_FILE.exists():
        return json.loads(PROGRESS_FILE.read_text())
    return {"done": [], "data": {}}


def save_progress(p: dict) -> None:
    PROGRESS_FILE.write_text(json.dumps(p, indent=2))

# ── Merge into parquet ────────────────────────────────────────────────────────

def merge_and_save(progress: dict) -> None:
    if not PARQUET.exists():
        print("⚠  market_caps.parquet not found.")
        return

    mc = pd.read_parquet(PARQUET)

    rows: dict[int, dict] = {}
    for ticker, year_caps in progress["data"].items():
        for yr_str, cap in year_caps.items():
            yr = int(yr_str)
            rows.setdefault(yr, {})[ticker] = cap

    if not rows:
        print("No data collected yet.")
        return

    new_df = pd.DataFrame(rows).T.sort_index()
    new_df.index = new_df.index.astype(int)

    for year in new_df.index:
        if year not in mc.index:
            mc.loc[year] = pd.NA
        for ticker in new_df.columns:
            val = new_df.loc[year, ticker]
            if pd.isna(val):
                continue
            if ticker not in mc.columns:
                mc[ticker] = pd.NA
            if pd.isna(mc.loc[year, ticker]):
                mc.loc[year, ticker] = val

    mc = mc.sort_index()
    mc.to_parquet(PARQUET)
    print(f"✅ Saved {PARQUET}")

    coverage = mc.notna().sum(axis=1)
    print("\nCoverage after merge:")
    for y in sorted(mc.index):
        if y <= 2014:
            print(f"  {y}: {int(coverage[y])} tickers")

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", metavar="TICKER", help="Test one ticker and exit")
    args = parser.parse_args()

    # Load prices for fast local price lookups
    prices_df = pd.read_parquet(PRICES_FILE) if PRICES_FILE.exists() else pd.DataFrame()
    if not prices_df.empty:
        prices_df.index = pd.to_datetime(prices_df.index)
        prices_df = prices_df.sort_index()

    # ── Test mode ─────────────────────────────────────────────────────────────
    if args.test:
        tk = args.test.upper()
        print(f"Testing {tk}...")
        caps = yf_market_caps(tk, prices_df)
        if not caps:
            print("No data returned — ticker may not have shares history this far back.")
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
        for val in comp.iloc[:, 1].dropna():
            tickers.update(str(val).split(","))
    if PARQUET.exists():
        tickers.update(pd.read_parquet(PARQUET).columns.tolist())

    tickers = sorted(t.strip() for t in tickers if t.strip() and "." not in t and len(t) <= 5)
    print(f"Total tickers: {len(tickers)}")

    # ── Load progress ─────────────────────────────────────────────────────────
    progress = load_progress()
    remaining = [t for t in tickers if t not in progress["done"]]
    pct = 100 * len(progress["done"]) / len(tickers) if tickers else 0
    print(f"Done: {len(progress['done'])} ({pct:.0f}%)  |  Remaining: {len(remaining)}\n")

    if not remaining:
        print("All tickers done — merging.")
        merge_and_save(progress)
        return

    # ── Fetch loop ────────────────────────────────────────────────────────────
    for i, ticker in enumerate(remaining):
        label = f"[{len(progress['done'])+1}/{len(tickers)}]"
        print(f"  {label} {ticker} ...", end=" ", flush=True)

        caps = yf_market_caps(ticker, prices_df)
        yrs  = sorted(caps.keys())
        print(f"{len(yrs)} years ({yrs[0]}–{yrs[-1]})" if yrs else "no data")

        progress["done"].append(ticker)
        if caps:
            progress["data"][ticker] = {str(y): v for y, v in caps.items()}

        # Save + merge every 50 tickers
        if (i + 1) % 50 == 0:
            save_progress(progress)
            merge_and_save(progress)
            print(f"  — checkpoint saved ({i+1} tickers this run)\n")

        time.sleep(SLEEP_SEC)

    save_progress(progress)
    merge_and_save(progress)
    print("\n🎉 All tickers fetched and merged!")
    PROGRESS_FILE.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
