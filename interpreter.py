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

SYSTEM_PROMPT = """You translate investment scenarios into a compact JSON config for an S&P 500 backtester.

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

━━━ RANK OPTIONS (pick exactly one) ━━━
  "return_best"            top performers / winners / best returns / highest gains
  "return_worst"           worst performers / losers / most beaten-down / contrarian
  "momentum_best"          trending / hot / breakout / relative strength / high RSI
  "market_cap_best"        largest / biggest / most valuable / blue chip / heavyweights
  "market_cap_growth_best" companies whose market cap grew most (not return — cap SIZE growth)

━━━ FILTERS ━━━
  field: market_cap | price | return
  op:    > < >= <=
  value: number + optional B/T/M suffix  (100B = $100 billion, 2.5T = $2.5 trillion)
  Examples: "market_cap > 100B"  "price > 50"  "return < 0"  "return > 0"
  Use [] when no filter.

  Cap tier shortcuts — always convert these when mentioned:
    mega-cap    → "market_cap > 200B"
    large-cap   → "market_cap > 10B"
    mid-cap     → "market_cap > 2B"  AND  "market_cap < 10B"
    small-cap   → "market_cap < 2B"
    micro-cap   → "market_cap < 300M"
    blue chip   → "market_cap > 50B"
    penny stock → "price < 5"

━━━ TIMING ━━━
  rebalance: how often to rotate — "annual" (default) | "quarterly" | "monthly"
  lookback:  how far back to measure performance — default "12m"
             Q1 strategy → "3m"; monthly → "1m"; semi-annual → "6m"
  hold:      months to hold before selling — null means hold until next rebalance
  buy_after_months: delay from Jan 1
    0 = buy in January (default)
    1 = buy in February
    3 = buy in April  ← use this for "after Q1 / April / wait for earnings"
    6 = buy in July   ← use this for "after Q2 / mid-year"

  Semi-annual strategies: use rebalance="annual", hold=6, buy_after_months=0

━━━ SPECIAL ━━━
  ticker: list of tickers for "hold X" or "just buy X" strategies
  index="SPY": equal-weight S&P 500 (use for: "buy the index", "just hold SPY", "passive")
  index="QQQ": equal-weight Nasdaq 100 (use for: "QQQ", "tech index", "Nasdaq 100")
  n: integer. Use "all" only for explicit "all/every/each stocks" phrasing.
     Default n=10 when user says "top stocks" without a number.

━━━ EXAMPLES ━━━

"buy 5 best performers each year"
{"rank":"return_best","n":5,"filters":[],"rebalance":"annual","lookback":"12m","hold":null,"buy_after_months":0,"ticker":null,"index":null}

"buy all losers from last year with market cap above $100B"
{"rank":"return_worst","n":"all","filters":["market_cap > 100B","return < 0"],"rebalance":"annual","lookback":"12m","hold":null,"buy_after_months":0,"ticker":null,"index":null}

"top 5 Q1 performers, hold 3 quarters"
{"rank":"return_best","n":5,"filters":[],"rebalance":"annual","lookback":"3m","hold":9,"buy_after_months":3,"ticker":null,"index":null}

"hold NVDA"
{"rank":"return_best","n":1,"filters":[],"rebalance":"annual","lookback":"12m","hold":null,"buy_after_months":0,"ticker":["NVDA"],"index":null}

"magnificent seven"
{"rank":"return_best","n":7,"filters":[],"rebalance":"annual","lookback":"12m","hold":null,"buy_after_months":0,"ticker":["AAPL","MSFT","NVDA","AMZN","META","GOOGL","TSLA"],"index":null}

"FAANG stocks"
{"rank":"return_best","n":5,"filters":[],"rebalance":"annual","lookback":"12m","hold":null,"buy_after_months":0,"ticker":["META","AAPL","AMZN","NFLX","GOOGL"],"index":null}

"top 10 momentum, large cap, quarterly"
{"rank":"momentum_best","n":10,"filters":["market_cap > 10B"],"rebalance":"quarterly","lookback":"6m","hold":null,"buy_after_months":0,"ticker":null,"index":null}

"buy top 5 losers with the highest market caps"
{"rank":"market_cap_best","n":5,"filters":["return < 0"],"rebalance":"annual","lookback":"12m","hold":null,"buy_after_months":0,"ticker":null,"index":null}

"buy the five biggest companies that fell last year"
{"rank":"market_cap_best","n":5,"filters":["return < 0"],"rebalance":"annual","lookback":"12m","hold":null,"buy_after_months":0,"ticker":null,"index":null}

"dogs of the dow — worst 10 large cap performers"
{"rank":"return_worst","n":10,"filters":["market_cap > 50B","return < 0"],"rebalance":"annual","lookback":"12m","hold":null,"buy_after_months":0,"ticker":null,"index":null}

"buy the dip — 20 most beaten down S&P stocks"
{"rank":"return_worst","n":20,"filters":[],"rebalance":"annual","lookback":"12m","hold":null,"buy_after_months":0,"ticker":null,"index":null}

"top 3 large cap stocks with negative returns, hold 6 months"
{"rank":"return_worst","n":3,"filters":["market_cap > 10B","return < 0"],"rebalance":"annual","lookback":"12m","hold":6,"buy_after_months":0,"ticker":null,"index":null}

"buy S&P 500 and hold"
{"rank":"return_best","n":1,"filters":[],"rebalance":"annual","lookback":"12m","hold":null,"buy_after_months":0,"ticker":null,"index":"SPY"}

"equal weight nasdaq 100"
{"rank":"return_best","n":1,"filters":[],"rebalance":"annual","lookback":"12m","hold":null,"buy_after_months":0,"ticker":null,"index":"QQQ"}

"buy top 10 Q1 winners hold for the rest of the year"
{"rank":"return_best","n":10,"filters":[],"rebalance":"annual","lookback":"3m","hold":9,"buy_after_months":3,"ticker":null,"index":null}

"rotate into top 5 momentum stocks every 3 months"
{"rank":"momentum_best","n":5,"filters":[],"rebalance":"quarterly","lookback":"3m","hold":null,"buy_after_months":0,"ticker":null,"index":null}

"small cap winners each month"
{"rank":"return_best","n":10,"filters":["market_cap < 2B"],"rebalance":"monthly","lookback":"1m","hold":null,"buy_after_months":0,"ticker":null,"index":null}

"companies that grew market cap the most last year, top 10"
{"rank":"market_cap_growth_best","n":10,"filters":[],"rebalance":"annual","lookback":"12m","hold":null,"buy_after_months":0,"ticker":null,"index":null}

"buy apple microsoft google amazon"
{"rank":"return_best","n":4,"filters":[],"rebalance":"annual","lookback":"12m","hold":null,"buy_after_months":0,"ticker":["AAPL","MSFT","GOOGL","AMZN"],"index":null}

"buy mid cap stocks with the best 6-month momentum, rotate quarterly"
{"rank":"momentum_best","n":10,"filters":["market_cap > 2B","market_cap < 10B"],"rebalance":"quarterly","lookback":"6m","hold":null,"buy_after_months":0,"ticker":null,"index":null}

"top 5 mega cap winners, buy after Q1 earnings"
{"rank":"return_best","n":5,"filters":["market_cap > 200B"],"rebalance":"annual","lookback":"3m","hold":null,"buy_after_months":3,"ticker":null,"index":null}

"worst 10 small cap stocks from last year, hold 6 months"
{"rank":"return_worst","n":10,"filters":["market_cap < 2B","return < 0"],"rebalance":"annual","lookback":"12m","hold":6,"buy_after_months":0,"ticker":null,"index":null}

"just hold SPY as benchmark"
{"rank":"return_best","n":1,"filters":[],"rebalance":"annual","lookback":"12m","hold":null,"buy_after_months":0,"ticker":null,"index":"SPY"}

"buy the 10 hottest stocks every month"
{"rank":"momentum_best","n":10,"filters":[],"rebalance":"monthly","lookback":"1m","hold":null,"buy_after_months":0,"ticker":null,"index":null}

"top 20 by market cap, annual rebalance"
{"rank":"market_cap_best","n":20,"filters":[],"rebalance":"annual","lookback":"12m","hold":null,"buy_after_months":0,"ticker":null,"index":null}

"buy 5 stocks with price under $20 that had best returns"
{"rank":"return_best","n":5,"filters":["price < 20"],"rebalance":"annual","lookback":"12m","hold":null,"buy_after_months":0,"ticker":null,"index":null}

━━━ DECISION RULES ━━━
1. "losers/fell/dropped/down/negative/beaten-down/contrarian/buy the dip" + "largest/biggest/highest market cap"
   → rank="market_cap_best"  +  filter "return < 0"

2. "losers/worst/fell/dropped/contrarian/beaten-down/buy the dip" (no market cap ranking)
   → rank="return_worst"

3. "market cap" or "largest/biggest/most valuable/blue chip" (no loser/winner modifier)
   → rank="market_cap_best"

4. "momentum/trending/hot/breakout/relative strength/high RSI/52-week high"
   → rank="momentum_best"  with lookback="6m" unless specified

5. "market cap grew/grew most/fastest growing" (cap SIZE increase, not stock return)
   → rank="market_cap_growth_best"

6. Always add "return < 0" when: losers/fell/dropped/down/negative/beaten-down/contrarian
7. Always add "return > 0" when: winners/gained/rose/positive/best  (only if rank is NOT return_best)

8. Timing shortcuts:
   "after Q1 / April / wait for Q1 earnings" → buy_after_months=3, lookback="3m"
   "after Q2 / mid-year / July"              → buy_after_months=6, lookback="6m"
   "semi-annual / every 6 months / twice a year" → rebalance="annual", hold=6
   "quarterly / every 3 months"              → rebalance="quarterly", lookback="3m"
   "monthly / every month"                   → rebalance="monthly",   lookback="1m"

9. Named portfolios:
   "FAANG"            → ticker=["META","AAPL","AMZN","NFLX","GOOGL"]
   "magnificent seven / mag 7" → ticker=["AAPL","MSFT","NVDA","AMZN","META","GOOGL","TSLA"]
   "dogs of the dow"  → rank="return_worst", n=10, filters=["market_cap > 50B","return < 0"]

10. When n is not specified and query is about "top stocks" → default n=10
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

# Best-known reliable free models on OpenRouter — tried first before dynamic list.
# DeepSeek R1 is excellent at structured JSON (think-blocks are already stripped).
# Llama 3.3 70B is the most consistent for instruction-following.
_PREFERRED_FREE_MODELS = [
    "deepseek/deepseek-r1:free",
    "meta-llama/llama-3.3-70b-instruct:free",
    "microsoft/phi-4:free",
    "qwen/qwq-32b:free",
    "google/gemini-2.0-flash-thinking-exp:free",
]


def _parse_params_b(model_id: str) -> float:
    m = re.search(r'(\d+(?:\.\d+)?)\s*b(?:\b|_|-)', model_id.lower())
    return float(m.group(1)) if m else 0.0


def _get_free_models(key: str) -> list:
    global _openrouter_free_models_cache
    # Preferred models always tried first; fill remainder from dynamic list.
    base = list(_PREFERRED_FREE_MODELS)
    if _openrouter_free_models_cache:
        extras = [m for m in _openrouter_free_models_cache if m not in base]
        return base + extras[:5]
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
            extras = [m for m in _openrouter_free_models_cache if m not in base]
            return base + extras[:5]
    except Exception:
        pass
    return base


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
                    "max_tokens": 400,
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


def _call_gemini(scenario: str, initial_capital: float, start_year: int) -> dict:
    import requests as _req
    key = os.environ["GEMINI_API_KEY"]
    user_msg = (
        f"Scenario: {scenario}\n"
        f"Initial capital: ${initial_capital:,.0f}\n"
        f"Start year: {start_year}\n\n"
        "Reply with raw JSON only."
    )
    # Auth keys (AQ.xxx, issued after May 2026) require x-goog-api-key header.
    # Standard keys (AIza...) also work with the header, so always use it.
    headers = {"x-goog-api-key": key, "Content-Type": "application/json"}

    errors = []

    # ── Attempt 1: new interactions API (simpler, supports auth keys) ──────
    try:
        r = _req.post(
            "https://generativelanguage.googleapis.com/v1beta/interactions",
            headers={**headers, "Api-Revision": "2026-05-20"},
            json={
                "model": "gemini-2.0-flash",
                "input": f"{SYSTEM_PROMPT}\n\n{user_msg}",
            },
            timeout=20,
        )
        if r.ok:
            data = r.json()
            raw = (
                data.get("interaction", {}).get("outputText")
                or data.get("outputText")
                or ""
            )
            raw = _extract_json(raw)
            if raw:
                return _compile(json.loads(raw), initial_capital, start_year)
            errors.append("interactions API: no JSON in response")
        else:
            errors.append(f"interactions API: HTTP {r.status_code} {r.text[:150]}")
    except Exception as e:
        errors.append(f"interactions API: {e}")

    # ── Attempt 2: classic generateContent endpoint with header auth ────────
    try:
        r = _req.post(
            "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent",
            headers=headers,
            json={
                "system_instruction": {"parts": [{"text": SYSTEM_PROMPT}]},
                "contents": [{"role": "user", "parts": [{"text": user_msg}]}],
                "generationConfig": {"temperature": 0, "maxOutputTokens": 400},
            },
            timeout=20,
        )
        if not r.ok:
            errors.append(f"generateContent: HTTP {r.status_code} {r.text[:150]}")
        else:
            try:
                raw = r.json()["candidates"][0]["content"]["parts"][0]["text"]
            except (KeyError, IndexError, TypeError) as e:
                errors.append(f"generateContent: malformed response ({e})")
                raise Exception("; ".join(errors))
            raw = _extract_json(raw)
            if not raw:
                errors.append("generateContent: no JSON in response")
            else:
                return _compile(json.loads(raw), initial_capital, start_year)
    except Exception as e:
        if "generateContent" not in str(e):
            errors.append(f"generateContent: {e}")

    raise Exception("; ".join(errors))


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
        max_tokens=400,
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
    "broadcom":"AVGO","eli lilly":"LLY","lilly":"LLY","unitedhealth":"UNH",
    "chevron":"CVX","home depot":"HD","costco":"COST","oracle":"ORCL",
    "abbvie":"ABBV","cocacola":"KO","coca cola":"KO","coke":"KO",
    "pepsi":"PEP","pepsico":"PEP","merck":"MRK","bank of america":"BAC",
    "goldman":"GS","morgan stanley":"MS","citigroup":"C","wells fargo":"WFC",
}

# Named portfolio shortcuts
_NAMED_PORTFOLIOS = {
    r'\b(magnificent.?seven|mag.?7)\b': ["AAPL","MSFT","NVDA","AMZN","META","GOOGL","TSLA"],
    r'\bfaang\b':                        ["META","AAPL","AMZN","NFLX","GOOGL"],
    r'\bfang\b':                         ["META","AMZN","NFLX","GOOGL"],
}

_SKIP_WORDS = {
    "I","A","AN","THE","AND","OR","IF","IN","ON","AT","TO","OF","MY",
    "BUY","SELL","GET","PUT","ALL","TOP","BOT","SP","ETF","YTD","QQ",
    "SPY","QQQ","USD","US","FOR","WITH","FROM","WHAT","HOW","IT","DO",
    "HOLD","BEST","LAST","YEAR","HIGH","LOW","NEXT","EACH","EVERY",
}

_WORD_NUMS = {
    "one":1,"two":2,"three":3,"four":4,"five":5,"six":6,"seven":7,
    "eight":8,"nine":9,"ten":10,"twelve":12,"fifteen":15,"twenty":20,
    "twenty five":25,"fifty":50,"hundred":100,
}


def _rule_based(scenario: str, initial_capital: float, start_year: int) -> dict:
    """Rule-based fallback — handles common patterns when LLM is unavailable."""
    text = scenario.lower().strip()

    simple = {
        "rank": "return_best",
        "n": 10,  # default 10, not 1
        "filters": [],
        "rebalance": "annual",
        "lookback": "12m",
        "hold": None,
        "buy_after_months": 0,
        "ticker": None,
        "index": None,
    }

    # ---- named portfolios (check before generic ticker scan) ----
    for pattern, tickers in _NAMED_PORTFOLIOS.items():
        if re.search(pattern, text):
            simple["ticker"] = tickers
            simple["n"] = len(tickers)
            return _compile(simple, initial_capital, start_year)

    # ---- index ----
    if "SPY" in scenario or re.search(
        r'\b(s&p.?500|sp500|s&p500|spy|index fund|passive|benchmark)\b', text
    ):
        simple["index"] = "SPY"
        return _compile(simple, initial_capital, start_year)
    if "QQQ" in scenario or re.search(r'\b(nasdaq.?100|qqq|tech.?index)\b', text):
        simple["index"] = "QQQ"
        return _compile(simple, initial_capital, start_year)

    # ---- named company → ticker ----
    tickers_found = []
    for name, ticker in _COMPANY_TO_TICKER.items():
        if name in text and ticker not in tickers_found:
            tickers_found.append(ticker)
    for t in re.findall(r'\b([A-Z]{2,5})\b', scenario):
        if t not in _SKIP_WORDS and t not in tickers_found:
            tickers_found.append(t)

    if re.search(r'\b(hold|held|just buy|just hold|invest in|only buy)\b', text) and tickers_found:
        simple["ticker"] = tickers_found
        simple["n"] = len(tickers_found)
        return _compile(simple, initial_capital, start_year)

    # ---- n ----
    n = None
    for pat in [
        r'\b(?:top|best|worst|bottom|largest|biggest)\s+(\d+)\b',
        r'\b(\d+)\s+(?:top|best|worst|bottom|largest|stocks?|companies|performers?|winners?|losers?)\b',
        r'\bbuy\s+(\d+)\b',
    ]:
        m = re.search(pat, text)
        if m:
            n = int(m.group(1))
            break
    if n is None:
        for word, val in _WORD_NUMS.items():
            if re.search(rf'\b(?:top|best|worst|bottom|largest|buy|pick)\s+{word}\b', text):
                n = val
                break
    # "all/every/each stocks" → n=999, but NOT "every month/quarter" (time words)
    if n is None and re.search(r'\b(all|every|each)\b', text) and not re.search(r'\b\d+\b', text):
        if not re.search(r'\b(month|quarter|year|week|day)\b', text):
            n = 999
    if n is not None:
        simple["n"] = n

    # ---- rank direction — most specific patterns first ----
    _has_loser = bool(re.search(
        r'\b(worst|bottom|loser|losers|contrarian|fell|dropped|declined|underperform|'
        r'dogs?|beaten.?down|buy.?the.?dip|dip|reversal|negative|laggard)\b', text
    ))
    _has_winner = bool(re.search(
        r'\b(winner|winners|best|top|gained|rose|positive|outperform)\b', text
    ))
    _has_cap_rank = bool(re.search(
        r'\b(market.?caps?|largest|biggest|most valuable|highest.{0,20}cap|'
        r'cap.{0,20}highest|blue.?chip|heavyweight|mega)\b', text
    ))
    _has_cap_growth = bool(re.search(
        r'\b(market.?cap.*(grew|grow|increas|gain|rose)|'
        r'(grew|grow|increas|gain|rose).*market.?cap|'
        r'fastest.?grow|cap.?growth)\b', text
    ))
    _has_momentum = bool(re.search(
        r'\b(momentum|trending|hot|breakout|relative strength|52.?week.?high|rsi|high.?flier)\b', text
    ))

    if _has_loser and _has_cap_rank:
        simple["rank"] = "market_cap_best"
    elif _has_cap_growth:
        simple["rank"] = "market_cap_growth_best"
    elif _has_momentum:
        simple["rank"] = "momentum_best"
        simple["lookback"] = "6m"
    elif _has_loser:
        simple["rank"] = "return_worst"
    elif _has_cap_rank:
        simple["rank"] = "market_cap_best"
    else:
        simple["rank"] = "return_best"

    # ---- rebalance frequency ----
    if re.search(r'\b(every month|monthly|each month|month.?to.?month)\b', text):
        simple["rebalance"] = "monthly"
        simple["lookback"] = "1m"
    elif re.search(r'\b(semi.?annual|every 6 months|twice a year|bi.?annual|biannual)\b', text):
        simple["hold"] = 6
    elif re.search(r'\b(every quarter|quarterly|each quarter|every 3 months|rotate.*quarter)\b', text):
        if not re.search(r'\b(every year|annually|each year)\b', text):
            simple["rebalance"] = "quarterly"
            simple["lookback"] = "3m"

    # ---- buy timing (Q1/Q2/month offset) ----
    if re.search(r'\b(after.?q1|after.?first.?quarter|april|wait.*q1.*earn)\b', text):
        simple["buy_after_months"] = 3
        if simple["lookback"] == "12m":
            simple["lookback"] = "3m"
    elif re.search(r'\b(after.?q2|after.?second.?quarter|july|mid.?year)\b', text):
        simple["buy_after_months"] = 6
        if simple["lookback"] == "12m":
            simple["lookback"] = "6m"
    elif re.search(r'\b(first quarter|q1)\b', text) and re.search(r'\b(hold|3 quarter|three quarter|rest.?of.?year)\b', text):
        simple["buy_after_months"] = 3
        simple["lookback"] = "3m"
        simple["hold"] = 9
    elif re.search(r'\b(january|jan\.?)\b', text) and re.search(r'\b(best|top|winner|feb)\b', text):
        simple["buy_after_months"] = 1
        simple["lookback"] = "1m"

    # ---- hold duration ----
    hm = re.search(r'\bhold(?:ing)?\s+(?:for\s+)?(\d+)\s*(month|quarter|year)', text)
    if hm:
        num, unit = int(hm.group(1)), hm.group(2)
        if unit.startswith("year"):
            simple["hold"] = num * 12
        elif unit.startswith("quarter"):
            simple["hold"] = num * 3
        else:
            simple["hold"] = num
    # "hold for the rest of the year" / "hold all year"
    elif re.search(r'\bhold.*(rest.?of.?year|all year|12 months?|a year|one year)\b', text):
        simple["hold"] = 12

    # ---- filters ----
    filters_raw = []
    mcap_m = re.search(
        r'market.?cap\s*(?:above|over|>|greater than|more than|of at least|exceeding)?\s*'
        r'\$?\s*(\d+(?:\.\d+)?)\s*(b(?:illion)?|t(?:rillion)?|m(?:illion)?)\b',
        text
    )
    if mcap_m:
        filters_raw.append(f"market_cap > {mcap_m.group(1)}{mcap_m.group(2)[0].upper()}")
    elif re.search(r'\bmega.?cap\b', text):
        filters_raw.append("market_cap > 200B")
    elif re.search(r'\blarge.?cap\b', text):
        filters_raw.append("market_cap > 10B")
    elif re.search(r'\bmid.?cap\b', text):
        filters_raw.append("market_cap > 2B")
        filters_raw.append("market_cap < 10B")
    elif re.search(r'\bsmall.?cap\b', text):
        filters_raw.append("market_cap < 2B")
    elif re.search(r'\bmicro.?cap\b', text):
        filters_raw.append("market_cap < 300M")
    elif re.search(r'\bblue.?chip\b', text):
        filters_raw.append("market_cap > 50B")

    # dogs of the dow — large cap losers
    if re.search(r'\bdogs?.*(dow|market)\b', text):
        if "market_cap > 50B" not in filters_raw:
            filters_raw.append("market_cap > 50B")

    price_m = re.search(r'(?:stock\s+)?price\s*(above|over|>|below|under|<)\s*\$?\s*(\d+(?:\.\d+)?)', text)
    if price_m:
        op = ">" if price_m.group(1) in ("above", "over", ">") else "<"
        filters_raw.append(f"price {op} {price_m.group(2)}")

    # penny stocks
    if re.search(r'\bpenny.?stocks?\b', text):
        filters_raw.append("price < 5")

    # return filter — always add when loser/winner language is explicit
    if _has_loser and "return < 0" not in filters_raw:
        # for "all losers" or when explicitly asking for negative-return stocks
        if simple["n"] >= 999 or re.search(
            r'\b(all|every|any|negative|fell|dropped|declined)\b', text
        ):
            filters_raw.append("return < 0")
    if _has_loser and simple["rank"] == "market_cap_best" and "return < 0" not in filters_raw:
        filters_raw.append("return < 0")

    simple["filters"] = filters_raw

    # ---- start year ----
    m = re.search(r'\b(20\d\d)\b', text)
    if m:
        yr = int(m.group(1))
        if 2001 <= yr <= 2025:
            simple["start_year"] = yr

    return _compile(simple, initial_capital, start_year)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def interpret(scenario: str, initial_capital: float = 10_000, start_year: int = 2001) -> dict:
    config = None
    error_msg = None

    if os.environ.get("GEMINI_API_KEY"):
        try:
            config = _call_gemini(scenario, initial_capital, start_year)
            config["_interpreter"] = "gemini"
        except Exception as e:
            error_msg = f"Gemini: {e}"

    if config is None and os.environ.get("OPENROUTER_API_KEY"):
        try:
            config = _call_openrouter(scenario, initial_capital, start_year)
            config["_interpreter"] = "openrouter"
        except Exception as e:
            error_msg = (error_msg or "") + f" | OpenRouter: {e}"

    if config is None and os.environ.get("GROQ_API_KEY"):
        try:
            config = _call_groq(scenario, initial_capital, start_year)
            config["_interpreter"] = "groq"
        except Exception as e:
            error_msg = (error_msg or "") + f" | Groq: {e}"

    if config is None and os.environ.get("ANTHROPIC_API_KEY"):
        try:
            config = _call_anthropic(scenario, initial_capital, start_year)
            config["_interpreter"] = "anthropic"
        except Exception as e:
            error_msg = (error_msg or "") + f" | Anthropic: {e}"

    if config is None:
        config = _rule_based(scenario, initial_capital, start_year)
        config["_interpreter"] = "rule-based"
        if error_msg:
            config["_interpreter_error"] = error_msg.lstrip(" | ")

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
