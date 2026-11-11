"""
Scenario interpreter for Strategy Lab.

Priority chain:
  1. OpenRouter (free models, dynamic) if OPENROUTER_API_KEY is set
  2. Groq API if GROQ_API_KEY is set
  3. Anthropic API if ANTHROPIC_API_KEY is set
  4. Rule-based fallback (last resort)

LLM path uses a compact 9-field schema — much simpler output = more reliable
on free models. A Python compiler expands it to the full engine config.
"""

import json
import os
import re
from typing import Optional


# ---------------------------------------------------------------------------
# Compact LLM schema — what we ask the LLM to produce
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You translate investment scenarios into a compact JSON config.

OUTPUT EXACTLY this JSON structure, no other text:
{
  "rank": "<see options>",
  "n": <integer or "all">,
  "filters": ["field op value", ...],
  "rebalance": "annual|quarterly|monthly",
  "lookback": "<N>m",
  "hold": <months as integer or null>,
  "buy_after_months": <0-11>,
  "ticker": null or ["TICK"],
  "index": null or "SPY" or "QQQ"
}

rank options (pick one):
  "return_best"          — buy top performers (winners/momentum)
  "return_worst"         — buy worst performers (contrarian/losers)
  "momentum_best"        — buy top momentum stocks
  "market_cap_best"      — buy largest companies
  "market_cap_growth_best" — buy companies whose market cap grew most

filters — list of strings, each: "field op value"
  field: market_cap | price | return
  op:    > < >= <=
  value: number with optional suffix B/T/M (e.g. 100B, 2.5T, 500M, 50)
  e.g. "market_cap > 100B", "price > 50", "return < 0"
  Use [] if no filter.

cap tier shortcuts (ALWAYS apply these as filters when mentioned):
  mega-cap  → market_cap > 200B
  large-cap → market_cap > 10B
  mid-cap   → market_cap > 2B AND market_cap < 10B
  small-cap → market_cap < 2B
  micro-cap → market_cap < 300M

n: how many stocks (integer). Use "all" only when user says all/every/each.

lookback: how far back to measure performance. Default "12m".
  "3m" = 3 months, "6m" = 6 months, "12m" = 1 year.

hold: months to hold before selling. null = full period (default).

buy_after_months: months after Jan 1 to buy. 0=January, 3=April (after Q1), 1=February.

ticker: only for "hold specific stock" strategies.
index: "SPY" for S&P 500 index, "QQQ" for Nasdaq 100.

EXAMPLES:
"buy 5 best performers each year"
{"rank":"return_best","n":5,"filters":[],"rebalance":"annual","lookback":"12m","hold":null,"buy_after_months":0,"ticker":null,"index":null}

"buy all losers from last year with market cap above $100B"
{"rank":"return_worst","n":"all","filters":["market_cap > 100B","return < 0"],"rebalance":"annual","lookback":"12m","hold":null,"buy_after_months":0,"ticker":null,"index":null}

"top 5 Q1 performers, hold 3 quarters"
{"rank":"return_best","n":5,"filters":[],"rebalance":"annual","lookback":"3m","hold":9,"buy_after_months":3,"ticker":null,"index":null}

"hold NVDA"
{"rank":"return_best","n":1,"filters":[],"rebalance":"annual","lookback":"12m","hold":null,"buy_after_months":0,"ticker":["NVDA"],"index":null}

"top 10 momentum, large cap, quarterly"
{"rank":"momentum_best","n":10,"filters":["market_cap > 10B"],"rebalance":"quarterly","lookback":"6m","hold":null,"buy_after_months":0,"ticker":null,"index":null}
"""


# ---------------------------------------------------------------------------
# Python compiler: simple schema → full engine config
# ---------------------------------------------------------------------------

def _parse_value(val_str: str) -> Optional[float]:
    """Parse '100B', '2.5T', '500M', '50' → float."""
    s = val_str.strip().upper().replace("$", "").replace(",", "")
    mult = 1.0
    if s.endswith("T"):
        mult = 1e12; s = s[:-1]
    elif s.endswith("B"):
        mult = 1e9; s = s[:-1]
    elif s.endswith("M"):
        mult = 1e6; s = s[:-1]
    elif s.endswith("K"):
        mult = 1e3; s = s[:-1]
    try:
        return float(s) * mult
    except ValueError:
        return None


def _parse_filter(f_str: str) -> Optional[dict]:
    """Parse 'market_cap > 100B' → {field, op, value}."""
    m = re.match(r'(\w+)\s*(>=|<=|>|<|=|!=)\s*(.+)', f_str.strip())
    if not m:
        return None
    field_raw, op, val_raw = m.group(1), m.group(2), m.group(3)
    field_map = {
        "market_cap": "market_cap",
        "marketcap": "market_cap",
        "price": "price",
        "return": "prior_return",
        "prior_return": "prior_return",
        "momentum": "prior_return",
    }
    field = field_map.get(field_raw.lower())
    if not field:
        return None
    value = _parse_value(val_raw)
    if value is None:
        return None
    return {"field": field, "op": op, "value": value}


def _parse_months(s) -> Optional[int]:
    """Parse '12m', '3m', '1y', '2q', 9 → int months. None stays None."""
    if s is None:
        return None
    if isinstance(s, (int, float)):
        return int(s)
    m = re.match(r'(\d+(?:\.\d+)?)\s*(m|mo|month|y|yr|year|q|quarter)', str(s).lower())
    if not m:
        try:
            return int(s)
        except (ValueError, TypeError):
            return None
    num, unit = float(m.group(1)), m.group(2)
    if unit.startswith("y"):
        return int(num * 12)
    if unit.startswith("q"):
        return int(num * 3)
    return int(num)


_RANK_TO_METHOD = {
    "return_best":           "top_n_prior_return",
    "return_worst":          "bottom_n_prior_return",
    "momentum_best":         "top_n_momentum",
    "momentum_worst":        "bottom_n_momentum",
    "market_cap_best":       "top_n_market_cap",
    "market_cap_growth_best":"top_n_market_cap_growth",
}


def _compile(simple: dict, initial_capital: float, start_year: int) -> dict:
    """Compile the compact LLM schema into a full engine config."""
    rank = simple.get("rank", "return_best")
    n_raw = simple.get("n", 1)
    n = 999 if str(n_raw).lower() == "all" else max(1, int(n_raw))

    raw_filters = simple.get("filters") or []
    filters = [pf for pf in (_parse_filter(f) for f in raw_filters) if pf]

    lookback = _parse_months(simple.get("lookback", "12m")) or 12
    hold = _parse_months(simple.get("hold"))  # None = full period

    index = simple.get("index")
    ticker = simple.get("ticker")

    if index == "SPY":
        method = "equal_weight_sp500"
    elif index == "QQQ":
        method = "equal_weight_nasdaq"
    elif ticker:
        method = "hold_ticker"
    else:
        method = _RANK_TO_METHOD.get(rank, "top_n_prior_return")

    rebalance = simple.get("rebalance", "annual")
    buy_after = int(simple.get("buy_after_months", 0) or 0)

    # Auto-generate display name
    n_str = "all" if n >= 999 else str(n)
    _rank_labels = {
        "return_best":            f"top {n_str} return",
        "return_worst":           f"bottom {n_str} return (contrarian)",
        "momentum_best":          f"top {n_str} momentum",
        "momentum_worst":         f"bottom {n_str} momentum (laggards)",
        "market_cap_best":        f"top {n_str} by market cap",
        "market_cap_growth_best": f"top {n_str} by market cap growth",
    }
    if index:
        base_name = f"Hold {index}"
    elif ticker:
        base_name = f"Hold {', '.join(ticker)}"
    else:
        base_name = _rank_labels.get(rank, rank)

    filter_suffix = ""
    for f_str in (raw_filters or []):
        filter_suffix += f", {f_str}"

    timing = ""
    if buy_after:
        import calendar
        mo = calendar.month_abbr[buy_after + 1] if buy_after < 12 else "Dec"
        timing = f", buy {mo}"
    if hold:
        timing += f", hold {hold}mo"

    display_name = f"{base_name} ({rebalance}){timing}{filter_suffix}"

    return {
        "selection_method": method,
        "n": n,
        "tickers": list(ticker) if ticker else [],
        "rebalance_frequency": rebalance,
        "selection_lookback_months": lookback,
        "buy_offset_months": buy_after,
        "hold_months": hold,
        "filters": filters,
        "start_year": start_year,
        "end_year": 2026,
        "initial_capital": initial_capital,
        "weight": "equal",
        "display_name": display_name,
    }


# ---------------------------------------------------------------------------
# JSON extraction helpers
# ---------------------------------------------------------------------------

def _extract_json(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"<reasoning>.*?</reasoning>", "", text, flags=re.DOTALL)
    text = re.sub(r"```(?:json)?", "", text).strip("`").strip()
    start = text.find("{")
    if start == -1:
        return ""
    depth = 0
    for i, ch in enumerate(text[start:], start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return ""


# ---------------------------------------------------------------------------
# OpenRouter
# ---------------------------------------------------------------------------

_openrouter_free_models_cache = []
_model_empty_strikes = {}


def _parse_params_b(model_id: str) -> float:
    m = re.search(r'(\d+(?:\.\d+)?)\s*b(?:\b|_|-)', model_id.lower())
    return float(m.group(1)) if m else 0.0


def _get_free_models(key: str) -> list:
    global _openrouter_free_models_cache
    if _openrouter_free_models_cache:
        return _openrouter_free_models_cache
    import requests as _req
    try:
        r = _req.get(
            "https://openrouter.ai/api/v1/models",
            headers={"Authorization": f"Bearer {key}"},
            timeout=10,
        )
        if r.ok:
            all_models = r.json().get("data", [])
            free = [
                m for m in all_models
                if m["id"].endswith(":free")
                and m.get("pricing", {}).get("prompt") == "0"
                and m.get("pricing", {}).get("completion") == "0"
                and "text" in m.get("architecture", {}).get("output_modalities", ["text"])
                and m.get("architecture", {}).get("modality") != "image->text"
            ]
            capable = [m for m in free if _parse_params_b(m["id"]) >= 3 or _parse_params_b(m["id"]) == 0]
            ranked = sorted(capable, key=lambda m: _parse_params_b(m["id"]), reverse=True)
            _openrouter_free_models_cache = [m["id"] for m in ranked[:10]]
            return _openrouter_free_models_cache
    except Exception:
        pass
    return []


def _call_openrouter(scenario: str, initial_capital: float, start_year: int) -> dict:
    import requests as _req
    key = os.environ["OPENROUTER_API_KEY"]
    user_msg = f"Scenario: {scenario}\nInitial capital: ${initial_capital:,.0f}\nStart year: {start_year}\n\nReply with raw JSON only."
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://strategy-lab.streamlit.app",
        "X-Title": "Strategy Lab",
    }
    all_models = _get_free_models(key)
    if not all_models:
        raise Exception("No free models available on OpenRouter right now — try again shortly.")
    models = [m for m in all_models if _model_empty_strikes.get(m, 0) < 2]
    if not models:
        models = all_models
    errors = []
    rate_limited = 0
    for model in models:
        try:
            r = _req.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                json={
                    "model": model,
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_msg},
                    ],
                    "temperature": 0,
                    "max_tokens": 250,
                },
                timeout=25,
            )
            if r.status_code == 429:
                rate_limited += 1
                errors.append(f"{model}: rate limited")
                continue
            if not r.ok:
                errors.append(f"{model}: HTTP {r.status_code}")
                continue
            try:
                raw = r.json()["choices"][0]["message"]["content"]
            except (KeyError, IndexError, TypeError):
                errors.append(f"{model}: malformed response")
                continue
            if not raw or not isinstance(raw, str):
                _model_empty_strikes[model] = _model_empty_strikes.get(model, 0) + 1
                errors.append(f"{model}: empty content")
                continue
            raw = _extract_json(raw)
            if not raw:
                _model_empty_strikes[model] = _model_empty_strikes.get(model, 0) + 1
                errors.append(f"{model}: no JSON found")
                continue
            _model_empty_strikes[model] = 0
            simple = json.loads(raw)
            return _compile(simple, initial_capital, start_year)
        except json.JSONDecodeError as e:
            errors.append(f"{model}: bad JSON ({e})")
        except Exception as e:
            errors.append(f"{model}: {e}")
    if rate_limited == len(models):
        global _openrouter_free_models_cache
        _openrouter_free_models_cache = []
    raise Exception("All models failed — " + "; ".join(errors[:5]))


def _call_groq(scenario: str, initial_capital: float, start_year: int) -> dict:
    from groq import Groq
    client = Groq(api_key=os.environ["GROQ_API_KEY"])
    user_msg = f"Scenario: {scenario}\nInitial capital: ${initial_capital:,.0f}\nStart year: {start_year}"
    resp = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        temperature=0,
        max_tokens=250,
    )
    raw = _extract_json(resp.choices[0].message.content.strip())
    simple = json.loads(raw)
    return _compile(simple, initial_capital, start_year)


def _call_anthropic(scenario: str, initial_capital: float, start_year: int) -> dict:
    import anthropic
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    user_msg = f"Scenario: {scenario}\nInitial capital: ${initial_capital:,.0f}\nStart year: {start_year}"
    msg = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=250,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_msg}],
    )
    raw = _extract_json(msg.content[0].text.strip())
    simple = json.loads(raw)
    return _compile(simple, initial_capital, start_year)


# ---------------------------------------------------------------------------
# Rule-based fallback — minimal, only for obvious dead-simple cases
# ---------------------------------------------------------------------------

_COMPANY_TO_TICKER = {
    "apple":"AAPL","microsoft":"MSFT","google":"GOOGL","alphabet":"GOOGL",
    "amazon":"AMZN","nvidia":"NVDA","meta":"META","facebook":"META",
    "tesla":"TSLA","netflix":"NFLX","amd":"AMD","intel":"INTC",
    "berkshire":"BRK-B","jpmorgan":"JPM","jp morgan":"JPM",
    "exxon":"XOM","walmart":"WMT","visa":"V","mastercard":"MA",
    "johnson":"JNJ","pfizer":"PFE","disney":"DIS","nike":"NKE",
    "salesforce":"CRM","uber":"UBER","airbnb":"ABNB","palantir":"PLTR",
}

_SKIP_WORDS = {
    "I","A","AN","THE","AND","OR","IF","IN","ON","AT","TO","OF","MY",
    "BUY","SELL","GET","PUT","ALL","TOP","BOT","SP","ETF","YTD","QQ",
    "SPY","QQQ","USD","US","FOR","WITH","FROM","WHAT","HOW","IT","DO",
}

_WORD_NUMS = {
    "one":1,"two":2,"three":3,"four":4,"five":5,"six":6,"seven":7,
    "eight":8,"nine":9,"ten":10,"fifteen":15,"twenty":20,"fifty":50,
}


def _rule_based(scenario: str, initial_capital: float, start_year: int) -> dict:
    """Minimal rule-based fallback — handles obvious patterns, defers rest to defaults."""
    text = scenario.lower().strip()

    simple = {
        "rank": "return_best",
        "n": 1,
        "filters": [],
        "rebalance": "annual",
        "lookback": "12m",
        "hold": None,
        "buy_after_months": 0,
        "ticker": None,
        "index": None,
    }

    # ---- index / specific ticker ----
    if "SPY" in scenario or re.search(r'\b(s&p.?500.?index|just.*spy|index fund)\b', text):
        simple["index"] = "SPY"
        return _compile(simple, initial_capital, start_year)
    if "QQQ" in scenario or re.search(r'\b(nasdaq.?100|qqq)\b', text):
        simple["index"] = "QQQ"
        return _compile(simple, initial_capital, start_year)

    tickers_found = []
    for name, ticker in _COMPANY_TO_TICKER.items():
        if name in text and ticker not in tickers_found:
            tickers_found.append(ticker)
    for t in re.findall(r'\b([A-Z]{2,5})\b', scenario):
        if t not in _SKIP_WORDS and t not in tickers_found:
            tickers_found.append(t)

    if re.search(r'\b(hold|held|just buy|just hold|invest in)\b', text) and tickers_found:
        simple["ticker"] = tickers_found
        simple["n"] = len(tickers_found)
        return _compile(simple, initial_capital, start_year)

    # ---- n ----
    n = None
    for pat in [
        r'\b(?:top|best|worst|bottom)\s+(\d+)\b',
        r'\b(\d+)\s+(?:top|best|worst|bottom|largest|stocks?|companies|performers?)\b',
    ]:
        m = re.search(pat, text)
        if m:
            n = int(m.group(1))
            break
    if n is None:
        for word, val in _WORD_NUMS.items():
            if re.search(rf'\b(?:top|best|worst|bottom|largest)\s+{word}\b', text):
                n = val
                break
    if n is None and re.search(r'\b(all|every|each)\b', text) and not re.search(r'\b\d+\b', text):
        n = 999
    simple["n"] = n or 1

    # ---- rank direction — check most specific first ----
    if re.search(r'\b(worst|bottom|loser|losers|contrarian|fell|dropped|declined|underperform|dogs?)\b', text):
        simple["rank"] = "return_worst"
    elif re.search(r'market.?cap.*(grew|grow|increas|gain|rose|biggest).*(year|prior|last)|'
                   r'(grew|grow|increas|gain|rose).*(market.?cap)', text):
        simple["rank"] = "market_cap_growth_best"
    elif re.search(r'\b(momentum|trending|hot|breakout|relative strength)\b', text):
        # momentum beats large-cap as the rank signal; large-cap becomes a filter below
        simple["rank"] = "momentum_best"
        simple["lookback"] = "6m"
    elif re.search(r'\b(market.?caps?|largest|biggest|most valuable)\b', text) \
            and not re.search(r'\b(best|top|winner|perform|return)\b', text):
        # only use market_cap as rank when there's no return/performance signal
        simple["rank"] = "market_cap_best"
    else:
        simple["rank"] = "return_best"

    # ---- rebalance frequency ----
    if re.search(r'\b(every month|monthly|each month)\b', text):
        simple["rebalance"] = "monthly"
        simple["lookback"] = "1m"
    elif re.search(r'\b(every quarter|quarterly|each quarter)\b', text):
        if not re.search(r'\b(every year|annually|each year)\b', text):
            simple["rebalance"] = "quarterly"
            simple["lookback"] = "3m"

    # ---- Q1/Q2 buy offset ----
    if re.search(r'\b(first quarter|q1)\b', text) and re.search(r'\b(hold|3 quarter|three quarter)\b', text):
        simple["buy_after_months"] = 3
        simple["lookback"] = "3m"
        simple["hold"] = 9
    elif re.search(r'\b(january|jan\.?)\b', text) and re.search(r'\b(best|top|winner|feb)\b', text):
        simple["buy_after_months"] = 1
        simple["lookback"] = "1m"

    # ---- hold duration ----
    hm = re.search(r'\bhold(?:ing)?\s+(?:for\s+)?(\d+)\s*(month|quarter)', text)
    if hm:
        num, unit = int(hm.group(1)), hm.group(2)
        simple["hold"] = num * 3 if unit.startswith("quarter") else num

    # ---- filters ----
    filters_raw = []
    mcap_m = re.search(
        r'market.?cap\s*(?:above|over|>|greater than|more than|of at least)?\s*\$?\s*(\d+(?:\.\d+)?)\s*(b(?:illion)?|t(?:rillion)?|m(?:illion)?)\b',
        text
    )
    if mcap_m:
        filters_raw.append(f"market_cap > {mcap_m.group(1)}{mcap_m.group(2)[0].upper()}")
    elif re.search(r'\bmega.?cap\b', text):
        filters_raw.append("market_cap > 100B")
    elif re.search(r'\blarge.?cap\b', text):
        filters_raw.append("market_cap > 10B")
    elif re.search(r'\bmid.?cap\b', text):
        filters_raw.append("market_cap > 2B")
        filters_raw.append("market_cap < 10B")
    elif re.search(r'\bsmall.?cap\b', text):
        filters_raw.append("market_cap < 2B")

    price_m = re.search(r'(?:stock\s+)?price\s*(above|over|>|below|under|<)\s*\$?\s*(\d+(?:\.\d+)?)', text)
    if price_m:
        op = ">" if price_m.group(1) in ("above", "over", ">") else "<"
        filters_raw.append(f"price {op} {price_m.group(2)}")

    # implicit return filter when "all losers/winners"
    if simple["n"] >= 999:
        if re.search(r'\b(loser|losers|lost|fell|dropped|declined|negative)\b', text):
            filters_raw.append("return < 0")
        elif re.search(r'\b(winner|winners|gained|rose|positive)\b', text):
            filters_raw.append("return > 0")

    simple["filters"] = filters_raw

    # ---- start year ----
    m = re.search(r'\b(20\d\d)\b', text)
    if m:
        yr = int(m.group(1))
        if 2001 <= yr <= 2025:
            simple["start_year"] = yr  # will be overridden by compile's start_year param anyway

    return _compile(simple, initial_capital, start_year)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def interpret(scenario: str, initial_capital: float = 10_000, start_year: int = 2001) -> dict:
    config = None
    error_msg = None

    if os.environ.get("OPENROUTER_API_KEY"):
        try:
            config = _call_openrouter(scenario, initial_capital, start_year)
        except Exception as e:
            error_msg = f"OpenRouter: {e}"

    if config is None and os.environ.get("GROQ_API_KEY"):
        try:
            config = _call_groq(scenario, initial_capital, start_year)
        except Exception as e:
            error_msg = f"Groq: {e}"

    if config is None and os.environ.get("ANTHROPIC_API_KEY"):
        try:
            config = _call_anthropic(scenario, initial_capital, start_year)
        except Exception as e:
            error_msg = f"Anthropic: {e}"

    if config is None:
        config = _rule_based(scenario, initial_capital, start_year)
        config["_interpreter"] = "rule-based"
        if error_msg:
            config["_interpreter_error"] = error_msg
    else:
        config["_interpreter"] = "llm"

    # Enforce required fields
    config.setdefault("selection_method", "top_n_prior_return")
    config.setdefault("n", 1)
    config.setdefault("tickers", [])
    config.setdefault("rebalance_frequency", "annual")
    config.setdefault("selection_lookback_months", 12)
    config.setdefault("buy_offset_months", 0)
    config.setdefault("filters", [])
    config.setdefault("end_year", 2026)
    config.setdefault("display_name", scenario[:80])
    config["initial_capital"] = initial_capital
    config["start_year"] = start_year

    return config
