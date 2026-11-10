"""
Data management and backtesting engine for Strategy Lab.
"""

import warnings
warnings.filterwarnings("ignore")

import json
import numpy as np
import pandas as pd
import requests
import yfinance as yf
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional, List, Dict

DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)

SP500_CSV_URL = (
    "https://raw.githubusercontent.com/fja05680/sp500/master/"
    "S%26P%20500%20Historical%20Components%20%26%20Changes%20(Updated).csv"
)

END_DATE = "2026-11-10"
START_DATE = "2000-01-01"


# ---------------------------------------------------------------------------
# Composition helpers
# ---------------------------------------------------------------------------

def _load_composition() -> pd.DataFrame:
    cache = DATA_DIR / "sp500_composition.csv"
    if cache.exists():
        return pd.read_csv(cache, index_col=0, parse_dates=True)
    resp = requests.get(SP500_CSV_URL, timeout=30)
    resp.raise_for_status()
    from io import StringIO
    df = pd.read_csv(StringIO(resp.text), index_col=0, parse_dates=True)
    df.to_csv(cache)
    return df


def _members_at(comp: pd.DataFrame, dt: pd.Timestamp) -> set:
    idx = comp.index[comp.index <= dt]
    if len(idx) == 0:
        return set()
    row_val = comp.iloc[comp.index.get_loc(idx[-1]), 0]
    return {t.strip() for t in str(row_val).split(",") if t.strip()}


def _members_full_year(comp: pd.DataFrame, year: int) -> set:
    start = pd.Timestamp(f"{year}-01-03")
    end = pd.Timestamp(f"{year}-12-28")
    return _members_at(comp, start) & _members_at(comp, end)


# ---------------------------------------------------------------------------
# DataManager
# ---------------------------------------------------------------------------

class DataManager:
    def __init__(self):
        self._comp: pd.DataFrame | None = None
        self._prices: pd.DataFrame | None = None

    # -- composition --

    def composition(self) -> pd.DataFrame:
        if self._comp is None:
            self._comp = _load_composition()
        return self._comp

    def members_year(self, year: int) -> set:
        return _members_full_year(self.composition(), year)

    def all_tickers(self) -> List[str]:
        comp = self.composition()
        seen: set[str] = set()
        for val in comp.iloc[:, 0]:
            for t in str(val).split(","):
                t = t.strip()
                if t:
                    seen.add(t)
        seen.update(["SPY", "QQQ"])
        return sorted(seen)

    # -- prices --

    def prices_loaded(self) -> bool:
        cache = DATA_DIR / "prices.parquet"
        return cache.exists() or self._prices is not None

    def _download_batch(self, tickers: List[str]) -> Dict[str, pd.Series]:
        collected: Dict[str, pd.Series] = {}
        batch_size = 50
        batches = [tickers[i : i + batch_size] for i in range(0, len(tickers), batch_size)]
        for batch in batches:
            try:
                raw = yf.download(
                    batch, start=START_DATE, end=END_DATE,
                    auto_adjust=True, progress=False, threads=True,
                )
                closes = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw
                for tk in batch:
                    if tk in closes.columns:
                        s = closes[tk].dropna()
                        if not s.empty:
                            collected[tk] = s
            except Exception:
                pass
        return collected

    def load_prices(self, progress_fn=None) -> "pd.DataFrame":
        if self._prices is not None:
            return self._prices  # already in memory, skip disk read

        cache = DATA_DIR / "prices.parquet"
        all_tickers = self.all_tickers()

        if cache.exists():
            existing = pd.read_parquet(cache)
            missing = [tk for tk in all_tickers if tk not in existing.columns]
            if missing:
                if progress_fn:
                    progress_fn(0, f"Filling {len(missing)} missing tickers…")
                new_data = self._download_batch(missing)
                if new_data:
                    combined = pd.concat([existing, pd.DataFrame(new_data)], axis=1)
                    combined.to_parquet(cache)
                    self._prices = combined
                else:
                    self._prices = existing
            else:
                self._prices = existing
            return self._prices

        if progress_fn:
            progress_fn(0, f"Downloading {len(all_tickers)} tickers (one-time setup)…")

        batches = [all_tickers[i : i + 50] for i in range(0, len(all_tickers), 50)]
        collected: Dict[str, pd.Series] = {}
        for i, batch in enumerate(batches):
            if progress_fn:
                progress_fn(i / len(batches), f"Batch {i+1}/{len(batches)}…")
            collected.update(self._download_batch(batch))

        df = pd.DataFrame(collected)
        df.to_parquet(cache)
        self._prices = df
        return df

    def prices(self) -> pd.DataFrame:
        if self._prices is None:
            cache = DATA_DIR / "prices.parquet"
            if cache.exists():
                self._prices = pd.read_parquet(cache)
            else:
                raise RuntimeError("Prices not loaded yet.")
        return self._prices

    # -- per-ticker helpers --

    def year_return(self, ticker: str, year: int) -> Optional[float]:
        p = self.prices()
        if ticker not in p.columns:
            return None
        s_start = f"{year}-01-01"
        s_end = "2026-11-09" if year == 2026 else f"{year}-12-31"
        series = p[ticker][s_start:s_end].dropna()
        if len(series) < 5:
            return None
        return float((series.iloc[-1] - series.iloc[0]) / series.iloc[0])

    def price_at(self, ticker: str, date: str) -> Optional[float]:
        p = self.prices()
        if ticker not in p.columns:
            return None
        slice_ = p[ticker][date:].dropna()
        if slice_.empty:
            return None
        return float(slice_.iloc[0])

    def shares_cache(self) -> Dict[str, float]:
        """Download shares outstanding for all tickers once and cache to JSON.

        Uses current shares via yfinance fast_info (reasonable for historical
        ranking since shares don't change dramatically for established companies).
        Downloads in parallel — ~60s for 1000 tickers, then instant from cache.
        """
        cache_file = DATA_DIR / "shares.json"
        if cache_file.exists():
            with open(cache_file) as f:
                return json.load(f)

        prices = self.prices()
        tickers = list(prices.columns)
        result: Dict[str, float] = {}

        def fetch_shares(tk: str) -> tuple:
            try:
                s = yf.Ticker(tk).fast_info.shares
                if s and float(s) > 0:
                    return tk, float(s)
            except Exception:
                pass
            return tk, None

        with ThreadPoolExecutor(max_workers=20) as ex:
            futures = {ex.submit(fetch_shares, tk): tk for tk in tickers}
            for fut in as_completed(futures):
                tk, shares = fut.result()
                if shares:
                    result[tk] = shares

        with open(cache_file, "w") as f:
            json.dump(result, f)

        return result

    def shares_loaded(self) -> bool:
        return (DATA_DIR / "shares.json").exists()

    def market_cap_proxy(self, tickers: List[str], year: int) -> Dict[str, float]:
        """Market cap = price at Jan 1 of year × shares outstanding (cached).

        Falls back to price-only if shares cache not yet downloaded.
        """
        shares = self.shares_cache() if self.shares_loaded() else {}
        out: Dict[str, float] = {}
        for tk in tickers:
            price = self.price_at(tk, f"{year}-01-01")
            if not price:
                continue
            if tk in shares:
                out[tk] = price * shares[tk]
            else:
                # fallback: price only (will be inaccurate for high-price/low-float stocks)
                out[tk] = price
        return out

    def january_return(self, ticker: str, year: int) -> Optional[float]:
        p = self.prices()
        if ticker not in p.columns:
            return None
        series = p[ticker][f"{year}-01-01":f"{year}-01-31"].dropna()
        if len(series) < 5:
            return None
        return float((series.iloc[-1] - series.iloc[0]) / series.iloc[0])

    def year_return_from_month(self, ticker: str, year: int, start_month: int) -> Optional[float]:
        p = self.prices()
        if ticker not in p.columns:
            return None
        start = f"{year}-{start_month:02d}-01"
        end = "2026-11-09" if year == 2026 else f"{year}-12-31"
        series = p[ticker][start:end].dropna()
        if len(series) < 5:
            return None
        return float((series.iloc[-1] - series.iloc[0]) / series.iloc[0])

    def period_momentum(self, tickers: List[str], year: int, months: int) -> Dict[str, float]:
        """Return over the prior `months` months ending at start of `year`."""
        p = self.prices()
        end_dt = pd.Timestamp(f"{year}-01-01")
        start_dt = end_dt - pd.DateOffset(months=months)
        out: dict[str, float] = {}
        for tk in tickers:
            if tk not in p.columns:
                continue
            series = p[tk][str(start_dt.date()) : str(end_dt.date())].dropna()
            if len(series) < 10:
                continue
            r = float((series.iloc[-1] - series.iloc[0]) / series.iloc[0])
            out[tk] = r
        return out


# ---------------------------------------------------------------------------
# Backtesting engine
# ---------------------------------------------------------------------------

# Tickers that represent the same underlying company (dual-class shares).
# Maps the secondary class → primary class. When iterating ranked lists,
# skip a ticker if its primary is already selected.
_SAME_COMPANY: Dict[str, str] = {
    "GOOG":  "GOOGL",   # Alphabet C → A
    "BRK-A": "BRK-B",   # Berkshire A → B
    "NWSA":  "NWS",     # News Corp A → B
    "FOXA":  "FOX",     # Fox Corp A → B
    "LSXMB": "LSXMA",   # Liberty SiriusXM
    "LSXMK": "LSXMA",
    "BATRK": "BATRA",   # Liberty Braves
    "FWONK": "FWONA",   # Liberty Formula One
}


def _dedup_tickers(ranked: list, n: int) -> list:
    """Pick top n tickers, skipping duplicates from the same company."""
    seen_companies: set = set()
    selected = []
    for tk in ranked:
        canonical = _SAME_COMPANY.get(tk, tk)
        if canonical in seen_companies:
            continue
        seen_companies.add(canonical)
        selected.append(tk)
        if len(selected) == n:
            break
    return selected


def run_backtest(config: dict, dm: DataManager) -> dict:
    """
    Execute a backtest from config.

    config keys:
      selection_method: str   — how to pick stocks each period
      n: int                  — number of stocks to hold (1 = concentrated)
      tickers: list[str]      — for hold_ticker method
      lookback_months: int    — for momentum methods (default 12)
      rebalance: "annual"     — only annual supported for now
      start_year: int
      end_year: int           — 2026 = YTD Nov 9
      initial_capital: float
      weight: "equal"         — equal-weight across selected tickers
    """
    method = config.get("selection_method", "top_n_prior_return")
    n = int(config.get("n", 1))
    start_year = int(config.get("start_year", 2002))
    end_year = int(config.get("end_year", 2026))
    capital = float(config.get("initial_capital", 10_000.0))
    lookback_months = int(config.get("lookback_months", 12))

    strategy_rows: List[dict] = []
    portfolio_value = capital

    for year in range(start_year, end_year + 1):
        # ---- select tickers ----
        if method == "hold_ticker":
            selected = config.get("tickers", ["SPY"])

        elif method in ("top_n_prior_return", "bottom_n_prior_return"):
            prior = year - 1
            universe = dm.members_year(prior)
            rets = {tk: dm.year_return(tk, prior) for tk in universe}
            rets = {k: v for k, v in rets.items() if v is not None}
            if not rets:
                strategy_rows.append(_empty_row(year, portfolio_value))
                continue
            reverse = method == "top_n_prior_return"
            ranked = [t for t, _ in sorted(rets.items(), key=lambda x: x[1], reverse=reverse)]
            selected = _dedup_tickers(ranked, n)

        elif method == "top_n_january_return":
            universe = dm.members_year(year)
            jan_rets = {tk: dm.january_return(tk, year) for tk in universe}
            jan_rets = {k: v for k, v in jan_rets.items() if v is not None}
            if not jan_rets:
                strategy_rows.append(_empty_row(year, portfolio_value))
                continue
            ranked = [t for t, _ in sorted(jan_rets.items(), key=lambda x: x[1], reverse=True)]
            selected = _dedup_tickers(ranked, n)

        elif method in ("top_n_momentum", "bottom_n_momentum"):
            universe = dm.members_year(year)
            moms = dm.period_momentum(list(universe), year, lookback_months)
            if not moms:
                strategy_rows.append(_empty_row(year, portfolio_value))
                continue
            reverse = method == "top_n_momentum"
            ranked = [t for t, _ in sorted(moms.items(), key=lambda x: x[1], reverse=reverse)]
            selected = _dedup_tickers(ranked, n)

        elif method == "top_n_market_cap":
            universe = dm.members_year(year)
            caps = dm.market_cap_proxy(list(universe), year)
            if not caps:
                strategy_rows.append(_empty_row(year, portfolio_value))
                continue
            ranked = [t for t, _ in sorted(caps.items(), key=lambda x: x[1], reverse=True)]
            selected = _dedup_tickers(ranked, n)

        elif method == "equal_weight_sp500":
            selected = ["SPY"]

        elif method == "equal_weight_nasdaq":
            selected = ["QQQ"]

        else:
            selected = ["SPY"]

        # ---- compute returns for selected tickers ----
        if method == "top_n_january_return":
            ticker_rets = [(tk, dm.year_return_from_month(tk, year, 2)) for tk in selected]
        else:
            ticker_rets = [(tk, dm.year_return(tk, year)) for tk in selected]
        ticker_rets = [(tk, r) for tk, r in ticker_rets if r is not None]

        if not ticker_rets:
            strategy_rows.append(_empty_row(year, portfolio_value))
            continue

        avg_ret = float(np.mean([r for _, r in ticker_rets]))
        start_val = portfolio_value
        portfolio_value *= 1 + avg_ret

        strategy_rows.append({
            "year": year,
            "tickers": [tk for tk, _ in ticker_rets],
            "ticker_returns": {tk: r for tk, r in ticker_rets},
            "portfolio_return": avg_ret,
            "start_value": start_val,
            "end_value": portfolio_value,
        })

    # ---- SPY benchmark (same period) ----
    spy_cap = float(config.get("initial_capital", 10_000.0))
    spy_rows: List[dict] = []
    for year in range(start_year, end_year + 1):
        r = dm.year_return("SPY", year)
        if r is not None:
            spy_cap *= 1 + r
            spy_rows.append({"year": year, "return": r, "value": spy_cap})

    init = float(config.get("initial_capital", 10_000.0))
    return {
        "strategy_rows": strategy_rows,
        "spy_rows": spy_rows,
        "final_value": portfolio_value,
        "spy_final": spy_cap,
        "total_roi": (portfolio_value - init) / init,
        "spy_roi": (spy_cap - init) / init,
        "initial_capital": init,
        "config": config,
    }


def _empty_row(year: int, value: float) -> dict:
    return {
        "year": year,
        "tickers": [],
        "ticker_returns": {},
        "portfolio_return": 0.0,
        "start_value": value,
        "end_value": value,
    }
