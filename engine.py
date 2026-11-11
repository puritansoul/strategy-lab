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

    # -- historical shares outstanding --

    _SHARES_YEARS = list(range(2001, 2027))

    def shares_history_loaded(self) -> bool:
        return (DATA_DIR / "shares_history.parquet").exists()

    def _shares_history_df(self) -> Optional[pd.DataFrame]:
        if not hasattr(self, "_sh_cache"):
            cache = DATA_DIR / "shares_history.parquet"
            self._sh_cache = pd.read_parquet(cache) if cache.exists() else None
        return self._sh_cache

    def load_shares_history(self, progress_fn=None) -> None:
        """Download year-start shares outstanding for every ticker via SEC EDGAR XBRL API.

        Uses CommonStockSharesOutstanding from 10-K/10-Q filings.
        No API key required. Rate-limited to 8 req/s to respect SEC guidelines.
        ~2 min on first run, then instant from cache.
        """
        cache = DATA_DIR / "shares_history.parquet"
        if cache.exists():
            return

        prices = self.prices()
        tickers = [tk for tk in prices.columns if tk not in ("SPY", "QQQ")]
        years = self._SHARES_YEARS

        # Step 1: fetch ticker → CIK mapping from SEC
        if progress_fn:
            progress_fn(0, "Fetching SEC ticker→CIK map…")
        try:
            r = requests.get(
                "https://www.sec.gov/files/company_tickers.json",
                headers={"User-Agent": "strategy-lab research@example.com"},
                timeout=30,
            )
            r.raise_for_status()
            cik_data = r.json()
            ticker_to_cik: Dict[str, int] = {
                v["ticker"].upper(): v["cik_str"]
                for v in cik_data.values()
            }
        except Exception:
            ticker_to_cik = {}

        result: Dict[str, Dict[int, float]] = {}
        total = len(tickers)
        done = 0
        import time as _time

        def fetch_one(tk: str):
            cik = ticker_to_cik.get(tk.upper())
            if not cik:
                return tk, None
            try:
                url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{int(cik):010d}.json"
                resp = requests.get(
                    url,
                    headers={"User-Agent": "strategy-lab research@example.com"},
                    timeout=20,
                )
                if not resp.ok:
                    return tk, None
                facts = resp.json()
                # CommonStockSharesOutstanding is the standard XBRL tag
                units = (
                    facts.get("facts", {})
                         .get("us-gaap", {})
                         .get("CommonStockSharesOutstanding", {})
                         .get("units", {})
                         .get("shares", [])
                )
                if not units:
                    return tk, None
                # Keep only annual (10-K) and quarterly (10-Q) filings, deduplicate by end date
                filings = [
                    (pd.Timestamp(u["end"]), float(u["val"]))
                    for u in units
                    if u.get("form") in ("10-K", "10-Q") and u.get("val", 0) > 0
                ]
                if not filings:
                    return tk, None
                # Sort and deduplicate — keep latest filing per end date
                seen: Dict[pd.Timestamp, float] = {}
                for dt, val in sorted(filings, key=lambda x: x[0]):
                    seen[dt] = val
                series = sorted(seen.items())  # [(Timestamp, val), ...]

                by_year: Dict[int, float] = {}
                for yr in years:
                    cutoff = pd.Timestamp(f"{yr}-01-01")
                    before = [(dt, v) for dt, v in series if dt <= cutoff]
                    if before:
                        by_year[yr] = before[-1][1]
                return tk, by_year if by_year else None
            except Exception:
                return tk, None

        # SEC rate limit: max 10 req/s — use 8 workers with small sleep
        with ThreadPoolExecutor(max_workers=8) as ex:
            futures = {ex.submit(fetch_one, tk): tk for tk in tickers}
            for fut in as_completed(futures):
                tk, by_year = fut.result()
                if by_year:
                    result[tk] = by_year
                done += 1
                if progress_fn and done % 25 == 0:
                    progress_fn(done / total, f"Shares history: {done}/{total} tickers…")
                _time.sleep(0.01)  # ~80 req/s peak → stays under SEC limit

        df = pd.DataFrame(result, index=years, dtype=float)
        df.to_parquet(cache)
        self._sh_cache = df

    def shares_cache(self) -> Dict[str, float]:
        """Snapshot shares (current) — used as fallback when history unavailable."""
        cache_file = DATA_DIR / "shares.json"
        if cache_file.exists():
            with open(cache_file) as f:
                return json.load(f)

        prices = self.prices()
        tickers = list(prices.columns)
        result: Dict[str, float] = {}

        def fetch_snap(tk: str) -> tuple:
            try:
                s = yf.Ticker(tk).fast_info.shares
                if s and float(s) > 0:
                    return tk, float(s)
            except Exception:
                pass
            return tk, None

        with ThreadPoolExecutor(max_workers=20) as ex:
            futures = {ex.submit(fetch_snap, tk): tk for tk in tickers}
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
        """Market cap = price × shares outstanding.

        Uses historical year-start shares (shares_history.parquet) when available,
        falls back to current snapshot (shares.json) for any gaps.
        Tickers with no shares data are omitted so market_cap filters treat them
        as unknown (excluded) rather than using a nonsense value.
        """
        date_str = f"{year}-01-01"
        hist = self._shares_history_df()
        snap = self.shares_cache() if self.shares_loaded() else {}

        out: Dict[str, float] = {}
        for tk in tickers:
            price = self.price_at(tk, date_str)
            if not price:
                continue

            shares = None
            # 1. Historical (accurate)
            if hist is not None and tk in hist.columns and year in hist.index:
                v = hist.loc[year, tk]
                # Sanity check: S&P 500 members always have ≥1M shares.
                # Values below this indicate the SEC filing used non-standard units
                # (e.g. reported in millions), which produces nonsense market caps.
                if pd.notna(v) and v >= 1_000_000:
                    shares = float(v)
            # 2. Snapshot fallback (less accurate for old years)
            if shares is None and tk in snap:
                s = snap[tk]
                if s >= 1_000_000:
                    shares = s

            if shares:
                out[tk] = price * shares
            # else omit — unknown market cap, filter will exclude
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

    def return_between(self, ticker: str, start_date: str, end_date: str) -> Optional[float]:
        p = self.prices()
        if ticker not in p.columns:
            return None
        series = p[ticker][start_date:end_date].dropna()
        if len(series) < 3:
            return None
        return float((series.iloc[-1] - series.iloc[0]) / series.iloc[0])

    def members_at_date(self, date: pd.Timestamp) -> set:
        return _members_at(self.composition(), date)

    def period_momentum(self, tickers: List[str], year: int, months: int) -> Dict[str, float]:
        p = self.prices()
        end_dt = pd.Timestamp(f"{year}-01-01")
        start_dt = end_dt - pd.DateOffset(months=months)
        out: Dict[str, float] = {}
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


def _compare(val, op: str, threshold: float) -> bool:
    if val is None:
        return False
    if op == ">":  return val > threshold
    if op == ">=": return val >= threshold
    if op == "<":  return val < threshold
    if op == "<=": return val <= threshold
    if op in ("=", "=="): return val == threshold
    if op == "!=": return val != threshold
    return True


def _apply_filters(
    universe: set,
    filters: list,
    dm: "DataManager",
    buy_date: pd.Timestamp,
    lookback_months: int,
) -> set:
    """Narrow the universe with WHERE-style filters before ranking."""
    if not filters:
        return universe

    remaining = set(universe)
    buy_str = str(buy_date.date())
    sel_start_str = str((buy_date - pd.DateOffset(months=lookback_months)).date())

    _caps = None
    _prices_map = None
    _returns_map = None

    for f in filters:
        field = f.get("field", "")
        op = f.get("op", ">")
        try:
            threshold = float(f.get("value", 0))
        except (TypeError, ValueError):
            continue

        if field == "market_cap":
            if _caps is None:
                _caps = dm.market_cap_proxy(list(remaining), buy_date.year)
            remaining = {tk for tk in remaining if _compare(_caps.get(tk), op, threshold)}

        elif field == "price":
            if _prices_map is None:
                _prices_map = {tk: dm.price_at(tk, buy_str) for tk in remaining}
            remaining = {tk for tk in remaining if _compare(_prices_map.get(tk), op, threshold)}

        elif field == "prior_return":
            if _returns_map is None:
                _returns_map = {
                    tk: dm.return_between(tk, sel_start_str, buy_str)
                    for tk in remaining
                }
            remaining = {tk for tk in remaining if _compare(_returns_map.get(tk), op, threshold)}

    return remaining


def _dedup_tickers(ranked: list, n: int) -> list:
    """Pick top n tickers, skipping duplicates from the same company.
    n=999 means take all (used when user says 'all losers/winners').
    """
    unlimited = n >= 999
    seen_companies: set = set()
    selected = []
    for tk in ranked:
        canonical = _SAME_COMPANY.get(tk, tk)
        if canonical in seen_companies:
            continue
        seen_companies.add(canonical)
        selected.append(tk)
        if not unlimited and len(selected) == n:
            break
    return selected


_MAX_DATE = pd.Timestamp("2026-11-09")


def _generate_periods(
    start_year: int,
    end_year: int,
    frequency: str,
    buy_offset_months: int = 0,
    hold_months: Optional[int] = None,
) -> List[Dict]:
    """Generate (buy, sell, year) dicts for each rebalance period.

    hold_months controls how long each position is held.
    If None, defaults to the natural period length (12/3/1 months).
    """
    # Default hold = natural period length
    _default_hold = {"annual": 12, "quarterly": 3, "monthly": 1}
    hold = hold_months if hold_months is not None else _default_hold.get(frequency or "annual", 12)

    periods: List[Dict] = []

    if frequency in ("annual", None, ""):
        for year in range(start_year, end_year + 1):
            buy = pd.Timestamp(f"{year}-01-01") + pd.DateOffset(months=buy_offset_months)
            if buy > _MAX_DATE:
                break
            sell = min(buy + pd.DateOffset(months=hold) - pd.Timedelta(days=1), _MAX_DATE)
            periods.append({"buy": buy, "sell": sell, "year": year})

    elif frequency == "quarterly":
        q_months = [1, 4, 7, 10]
        for year in range(start_year, end_year + 1):
            for i, qm in enumerate(q_months):
                buy = pd.Timestamp(f"{year}-{qm:02d}-01") + pd.DateOffset(months=buy_offset_months)
                if buy > _MAX_DATE:
                    break
                sell = min(buy + pd.DateOffset(months=hold) - pd.Timedelta(days=1), _MAX_DATE)
                if buy >= sell:
                    continue
                periods.append({"buy": buy, "sell": sell, "year": year, "quarter": i + 1})

    elif frequency == "monthly":
        for year in range(start_year, end_year + 1):
            for month in range(1, 13):
                buy = pd.Timestamp(f"{year}-{month:02d}-01") + pd.DateOffset(months=buy_offset_months)
                if buy > _MAX_DATE:
                    break
                sell = min(buy + pd.DateOffset(months=hold) - pd.Timedelta(days=1), _MAX_DATE)
                if buy >= sell:
                    continue
                periods.append({"buy": buy, "sell": sell, "year": year, "month": month})

    return periods


def _aggregate_to_annual(period_rows: List[dict]) -> List[dict]:
    by_year: Dict[int, dict] = {}
    for row in period_rows:
        year = row["year"]
        if year not in by_year:
            by_year[year] = {
                "year": year,
                "tickers": set(),
                "ticker_returns": {},
                "start_value": row["start_value"],
                "end_value": row["end_value"],
                "buy_date": row.get("buy"),
                "sell_date": row.get("sell"),
            }
        else:
            by_year[year]["end_value"] = row["end_value"]
            by_year[year]["sell_date"] = row.get("sell")  # last period's sell
        by_year[year]["tickers"].update(row.get("tickers", []))
        by_year[year]["ticker_returns"].update(row.get("ticker_returns", {}))

    result = []
    for year in sorted(by_year.keys()):
        d = by_year[year]
        sv, ev = d["start_value"], d["end_value"]
        result.append({
            "year": year,
            "tickers": sorted(d["tickers"]),
            "ticker_returns": d["ticker_returns"],
            "portfolio_return": (ev - sv) / sv if sv > 0 else 0.0,
            "start_value": sv,
            "end_value": ev,
            "buy_date": d.get("buy_date"),
            "sell_date": d.get("sell_date"),
        })
    return result


def run_backtest(config: dict, dm: DataManager) -> dict:
    method = config.get("selection_method", "top_n_prior_return")
    n = int(config.get("n", 1))
    start_year = int(config.get("start_year", 2002))
    end_year = int(config.get("end_year", 2026))
    capital = float(config.get("initial_capital", 10_000.0))
    frequency = config.get("rebalance_frequency", "annual")
    buy_offset = int(config.get("buy_offset_months", 0))
    hold_months = config.get("hold_months")  # None = natural period length

    # Default lookback: 6 months for momentum methods, 12 for everything else
    default_lb = 6 if method in ("top_n_momentum", "bottom_n_momentum") else 12
    lookback_months = int(config.get("selection_lookback_months", config.get("lookback_months", default_lb)))

    periods = _generate_periods(start_year, end_year, frequency, buy_offset, hold_months)

    portfolio_value = capital
    period_rows: List[dict] = []

    for period in periods:
        buy_date = period["buy"]
        sell_date = period["sell"]
        year = period["year"]
        buy_str = str(buy_date.date())
        sell_str = str(sell_date.date())
        sel_end_str = buy_str
        sel_start_str = str((buy_date - pd.DateOffset(months=lookback_months)).date())

        universe = dm.members_at_date(buy_date)

        # ---- apply WHERE filters ----
        filters = config.get("filters", [])
        if filters:
            universe = _apply_filters(universe, filters, dm, buy_date, lookback_months)

        # ---- select tickers ----
        if method == "hold_ticker":
            selected = config.get("tickers", ["SPY"])

        elif method in ("top_n_prior_return", "bottom_n_prior_return",
                        "top_n_momentum", "bottom_n_momentum"):
            rets = {tk: dm.return_between(tk, sel_start_str, sel_end_str) for tk in universe}
            rets = {k: v for k, v in rets.items() if v is not None}
            if not rets:
                period_rows.append(_empty_period_row(period, portfolio_value))
                continue
            reverse = method in ("top_n_prior_return", "top_n_momentum")
            ranked = [t for t, _ in sorted(rets.items(), key=lambda x: x[1], reverse=reverse)]
            selected = _dedup_tickers(ranked, n)

        elif method == "top_n_market_cap":
            caps = dm.market_cap_proxy(list(universe), year)
            if not caps:
                period_rows.append(_empty_period_row(period, portfolio_value))
                continue
            ranked = [t for t, _ in sorted(caps.items(), key=lambda x: x[1], reverse=True)]
            selected = _dedup_tickers(ranked, n)

        elif method == "top_n_market_cap_growth":
            # Rank by % change in market cap over the lookback window
            lookback_year = buy_date.year - max(1, lookback_months // 12)
            caps_now = dm.market_cap_proxy(list(universe), buy_date.year)
            caps_then = dm.market_cap_proxy(list(universe), lookback_year)
            growth = {
                tk: (caps_now[tk] - caps_then[tk]) / caps_then[tk]
                for tk in caps_now
                if tk in caps_then and caps_then[tk] > 0
            }
            if not growth:
                period_rows.append(_empty_period_row(period, portfolio_value))
                continue
            ranked = [t for t, _ in sorted(growth.items(), key=lambda x: x[1], reverse=True)]
            selected = _dedup_tickers(ranked, n)

        elif method == "equal_weight_sp500":
            selected = ["SPY"]

        elif method == "equal_weight_nasdaq":
            selected = ["QQQ"]

        else:
            selected = ["SPY"]

        # ---- holding period returns ----
        ticker_rets = [(tk, dm.return_between(tk, buy_str, sell_str)) for tk in selected]
        ticker_rets = [(tk, r) for tk, r in ticker_rets if r is not None]

        if not ticker_rets:
            period_rows.append(_empty_period_row(period, portfolio_value))
            continue

        avg_ret = float(np.mean([r for _, r in ticker_rets]))
        start_val = portfolio_value
        portfolio_value *= 1 + avg_ret

        period_rows.append({
            **period,
            "tickers": [tk for tk, _ in ticker_rets],
            "ticker_returns": {tk: r for tk, r in ticker_rets},
            "period_return": avg_ret,
            "start_value": start_val,
            "end_value": portfolio_value,
        })

    strategy_rows = _aggregate_to_annual(period_rows)

    # ---- SPY benchmark ----
    spy_cap = capital
    spy_rows: List[dict] = []
    for year in range(start_year, end_year + 1):
        r = dm.year_return("SPY", year)
        if r is not None:
            spy_cap *= 1 + r
            spy_rows.append({"year": year, "return": r, "value": spy_cap})

    init = capital
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


def _empty_period_row(period: dict, value: float) -> dict:
    return {
        **period,
        "tickers": [],
        "ticker_returns": {},
        "period_return": 0.0,
        "start_value": value,
        "end_value": value,
    }
