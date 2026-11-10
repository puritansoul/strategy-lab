"""
Scenario interpreter for Strategy Lab.

Priority chain:
  1. Google Gemini (free — gemini-1.5-flash) if GEMINI_API_KEY is set
  2. Groq API (free — llama-3.3-70b) if GROQ_API_KEY is set
  3. Anthropic API if ANTHROPIC_API_KEY is set
  4. Rule-based keyword parser (no key required, always available as fallback)
"""

import json
import os
import re
from typing import Optional


# ---------------------------------------------------------------------------
# Shared LLM prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a financial backtesting assistant.
The user will describe an investment scenario in plain English.
Your job is to translate it into a structured JSON config for a backtesting engine.

SELECTION METHODS:
- "top_n_prior_return"    : buy top N stocks by return over the lookback window (winners/momentum)
- "bottom_n_prior_return" : buy bottom N stocks by return over the lookback window (contrarian)
- "top_n_momentum"        : buy top N stocks by recent momentum (default 6-month lookback)
- "bottom_n_momentum"     : buy worst N stocks by recent momentum
- "top_n_market_cap"      : buy top N stocks by market cap (largest companies)
- "hold_ticker"           : simply hold specific named ticker(s) every period
- "equal_weight_sp500"    : hold SPY (S&P 500 index fund)
- "equal_weight_nasdaq"   : hold QQQ (Nasdaq 100 index fund)

PARAMETERS:
- n: number of stocks to hold (1 = concentrated, 5/10/20 = diversified)
- tickers: list of ticker symbols — ONLY for hold_ticker method
- rebalance_frequency: "annual" (default), "quarterly", or "monthly"
- selection_lookback_months: months to look back for ranking (default: 12 annual, 3 quarterly, 1 monthly)
- buy_offset_months: months to delay buying after lookback ends (default 0; use 1 for January effect)
- start_year: first year to trade (2001-2024). Default 2001.
- end_year: last year (use 2026 for current). Default 2026.
- initial_capital: starting $ (default 10000)
- weight: always "equal"

RULES:
- "best", "top", "winners", "best performing" → top_n_prior_return
- "worst", "bottom", "losers", "contrarian" → bottom_n_prior_return
- "market cap", "market caps", "largest", "biggest", "most valuable" → top_n_market_cap
- "momentum", "trending", "hot", "breakout" → top_n_momentum
- "hold X", "buy X", "invest in X", "just X" (specific company) → hold_ticker
- "S&P 500 index", "just SPY", "index fund" → equal_weight_sp500
- "January effect", "best in January", "buy in February based on January" → top_n_prior_return + buy_offset_months=1 + selection_lookback_months=1
- "first quarter", "Q1 performers", "buy in April based on Q1" → top_n_prior_return + buy_offset_months=3 + selection_lookback_months=3 + rebalance_frequency=annual
- "second quarter", "Q2 performers" → top_n_prior_return + buy_offset_months=6 + selection_lookback_months=3 + rebalance_frequency=annual
- "every quarter", "quarterly rotation", "rebalance quarterly" → rebalance_frequency=quarterly, selection_lookback_months=3
- "every month", "monthly rotation", "rebalance monthly" → rebalance_frequency=monthly, selection_lookback_months=1
- If user says "top 5" or "5 stocks" or "best 5" → n=5
- If no number mentioned → n=1
- If start year not mentioned → use 2001
- NOTE: "hold for 3 quarters" with "rebalance every year" means buy_offset_months=3 (buy in April after Q1 ends), NOT quarterly rebalancing

IMPORTANT: Your entire response must be a single valid JSON object. No explanation, no markdown, no code fences, no extra text — just the raw JSON object starting with { and ending with }.

Example — "buy top 5 Q1 performers, hold 3 quarters, rebalance annually":
{"selection_method":"top_n_prior_return","n":5,"tickers":[],"rebalance_frequency":"annual","selection_lookback_months":3,"buy_offset_months":3,"start_year":2001,"end_year":2026,"initial_capital":10000,"weight":"equal","display_name":"Top 5 Q1 winners, hold Apr–Dec (annual)"}

Example — default annual:
{"selection_method":"top_n_prior_return","n":5,"tickers":[],"rebalance_frequency":"annual","selection_lookback_months":12,"buy_offset_months":0,"start_year":2001,"end_year":2026,"initial_capital":10000,"weight":"equal","display_name":"Top 5 prior-year winners (annual)"}
"""


_openrouter_free_models_cache = []


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
            available = {m["id"] for m in all_models if m["id"].endswith(":free")}

            # Preferred order: largest/most capable instruction models first
            # Skip vision-only and tiny (<3B) models
            _priority = [
                "nvidia/nemotron-3-ultra-550b-a55b:free",
                "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free",
                "google/gemma-4-31b-it:free",
                "google/gemma-4-26b-a4b-it:free",
                "nvidia/nemotron-3.5-lightning:free",
                "xingchen-agi/xing4.0-29b-a4b:free",
                "cohere/north-mini-code:free",
            ]
            _skip_any = ("vl", "embed", "rerank", "1.7b", "2.6b", "note")
            ordered = [m for m in _priority if m in available]
            # append any new free models not in our priority list
            extras = [
                m for m in available
                if m not in _priority
                and not any(s in m.lower() for s in _skip_any)
            ]
            ordered.extend(sorted(extras))
            _openrouter_free_models_cache = ordered[:10]
            return _openrouter_free_models_cache
    except Exception:
        pass
    return ["meta-llama/llama-3.1-8b-instruct:free"]


def _call_openrouter(scenario: str, initial_capital: float, start_year: int) -> dict:
    import requests as _req
    key = os.environ["OPENROUTER_API_KEY"]
    user_msg = f"Scenario: {scenario}\nInitial capital: ${initial_capital:,.0f}\nStart year: {start_year}"
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://strategy-lab.streamlit.app",
        "X-Title": "Strategy Lab",
    }
    models = _get_free_models(key)
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
                        {"role": "user", "content": user_msg + "\n\nReply with raw JSON only."},
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
                errors.append(f"{model}: empty content")
                continue
            raw = re.sub(r"```(?:json)?", "", raw.strip()).strip("`").strip()
            return json.loads(raw)
        except json.JSONDecodeError as e:
            errors.append(f"{model}: bad JSON ({e})")
        except Exception as e:
            errors.append(f"{model}: {e}")
    # if everything was rate-limited, clear cache so next call re-fetches fresh model list
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
        max_tokens=300,
    )
    raw = resp.choices[0].message.content.strip()
    raw = re.sub(r"```(?:json)?", "", raw).strip("`").strip()
    return json.loads(raw)


def _call_anthropic(scenario: str, initial_capital: float, start_year: int) -> dict:
    import anthropic
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    user_msg = f"Scenario: {scenario}\nInitial capital: ${initial_capital:,.0f}\nStart year: {start_year}"
    msg = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=300,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_msg}],
    )
    raw = msg.content[0].text.strip()
    raw = re.sub(r"```(?:json)?", "", raw).strip("`").strip()
    return json.loads(raw)


# ---------------------------------------------------------------------------
# Rule-based fallback
# ---------------------------------------------------------------------------

_WORD_NUMS = {
    "one":1,"two":2,"three":3,"four":4,"five":5,
    "six":6,"seven":7,"eight":8,"nine":9,"ten":10,
    "fifteen":15,"twenty":20,"twenty-five":25,"fifty":50,
}

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


def _rule_based(scenario: str, initial_capital: float, start_year: int) -> dict:
    text = scenario.lower().strip()
    config = {
        "selection_method": "top_n_prior_return",
        "n": 1,
        "tickers": [],
        "rebalance_frequency": "annual",
        "selection_lookback_months": 12,
        "buy_offset_months": 0,
        "start_year": start_year,
        "end_year": 2026,
        "initial_capital": initial_capital,
        "weight": "equal",
        "display_name": scenario[:80],
    }

    # ---- extract N ----
    _n_pats = [
        r'\b(?:top|best|worst|bottom)\s+(\d+)\b',
        r'\b(\d+)\s+(?:top|best|worst|bottom)\b',
        r'\b(\d+)\s+largest\b',
        r'\b(\d+)\s+stocks?\b',
        r'\b(\d+)\s+companies\b',
        r'\b(\d+)\s+performers?\b',
        r'\b(\d+)\s+positions?\b',
        r'\b(\d+)\s+holdings?\b',
        r'\bperforming\s+(\d+)\b',
        r'\b(\d+)\s+performing\b',
    ]
    n = 1
    for pat in _n_pats:
        m = re.search(pat, text)
        if m:
            n = int(m.group(1))
            break
    if n == 1:
        for word, val in _WORD_NUMS.items():
            if re.search(rf'\b(?:top|best|worst|bottom|largest)\s+{word}\b', text):
                n = val
                break
    config["n"] = n

    # ---- detect tickers / company names ----
    tickers_found = []
    for name, ticker in _COMPANY_TO_TICKER.items():
        if name in text and ticker not in tickers_found:
            tickers_found.append(ticker)
    for t in re.findall(r'\b([A-Z]{1,5})\b', scenario):
        if t not in _SKIP_WORDS and t not in tickers_found:
            tickers_found.append(t)

    # ---- rebalance frequency ----
    if re.search(r'\b(every month|monthly|each month|per month|month by month)\b', text):
        config["rebalance_frequency"] = "monthly"
        config["selection_lookback_months"] = 1
    elif re.search(r'\b(every quarter|quarterly|each quarter|per quarter|quarter by quarter)\b', text):
        # "rebalance every year" overrides "quarterly" if both appear
        if not re.search(r'\b(every year|annually|each year|rebalanc.*year|year.*rebalanc)\b', text):
            config["rebalance_frequency"] = "quarterly"
            config["selection_lookback_months"] = 3

    # ---- Quarter-based buy offset (annual rebalance, buy after a specific quarter) ----
    # e.g. "top performers from first quarter, hold 3 quarters" → buy April, annual
    if re.search(r'\b(first quarter|q1|quarter one)\b', text) and re.search(r'\b(hold|3 quarter|three quarter)\b', text):
        config["rebalance_frequency"] = "annual"
        config["selection_lookback_months"] = 3
        config["buy_offset_months"] = 3
    elif re.search(r'\b(second quarter|q2|quarter two)\b', text) and re.search(r'\b(hold|2 quarter|two quarter)\b', text):
        config["rebalance_frequency"] = "annual"
        config["selection_lookback_months"] = 3
        config["buy_offset_months"] = 6

    # ---- January effect (detect before general method) ----
    if re.search(r'\b(january|jan\.?)\b', text) and re.search(r'\b(best|top|perform|winner|feb|february)\b', text):
        config["selection_method"] = "top_n_prior_return"
        config["buy_offset_months"] = 1
        config["selection_lookback_months"] = 1
    # ---- general selection method ----
    elif re.search(r'\b(hold|held|keep|just buy|just hold|bought|invest in|put.*into)\b', text) and tickers_found:
        config["selection_method"] = "hold_ticker"
        config["tickers"] = tickers_found
        config["n"] = len(tickers_found)
    elif "SPY" in scenario or re.search(r'\b(s&p\s*500\s*index|just.*index|index fund)\b', text):
        config["selection_method"] = "equal_weight_sp500"
    elif "QQQ" in scenario or re.search(r'\b(nasdaq\s*100|qqq)\b', text):
        config["selection_method"] = "equal_weight_nasdaq"
    elif re.search(r'\b(worst|bottom|loser|losing|losers|loss|losses|contrarian|fallen|fell|dropped|declined|underperform|mean.?revert|bad|poor|laggard|laggards|dog|dogs)\b', text):
        config["selection_method"] = "bottom_n_prior_return"
    elif re.search(r'\b(market.?caps?|largest|biggest|most valuable|mega.?cap)\b', text):
        config["selection_method"] = "top_n_market_cap"
    elif re.search(r'\b(momentum|trending|hot|breakout|relative strength)\b', text):
        config["selection_method"] = "top_n_momentum"
        if re.search(r'\b(3.?month|quarter|90.?day)\b', text):
            config["selection_lookback_months"] = 3
        elif re.search(r'\b(6.?month|half.?year|180.?day)\b', text):
            config["selection_lookback_months"] = 6
        else:
            config["selection_lookback_months"] = 6
    else:
        config["selection_method"] = "top_n_prior_return"

    # ---- detect start year ----
    m = re.search(r'\b(20\d\d)\b', text)
    if m:
        yr = int(m.group(1))
        if 2001 <= yr <= 2025:
            config["start_year"] = yr

    # ---- display name ----
    freq = config.get("rebalance_frequency", "annual")
    lb = config.get("selection_lookback_months", 12)
    offset = config.get("buy_offset_months", 0)
    freq_label = {"annual": "annual", "quarterly": "quarterly", "monthly": "monthly"}.get(freq, freq)
    jan_effect = (offset == 1 and lb == 1)
    q1_effect = (offset == 3 and lb == 3 and freq == "annual")
    _labels = {
        "top_n_prior_return":    (f"Top {n} January winners (buy Feb)" if jan_effect
                                  else f"Top {n} Q1 winners (buy Apr, hold 3Q)" if q1_effect
                                  else f"Top {n} prior-{lb}mo winner{'s' if n>1 else ''} ({freq_label})"),
        "bottom_n_prior_return": f"Bottom {n} prior-{lb}mo loser{'s' if n>1 else ''} ({freq_label})",
        "top_n_momentum":        f"Top {n} momentum stock{'s' if n>1 else ''} ({lb}m, {freq_label})",
        "bottom_n_momentum":     f"Bottom {n} momentum stock{'s' if n>1 else ''} ({lb}m, {freq_label})",
        "top_n_market_cap":      f"Top {n} largest by market cap ({freq_label})",
        "hold_ticker":           f"Hold {', '.join(config['tickers'])}",
        "equal_weight_sp500":    "S&P 500 index (SPY)",
        "equal_weight_nasdaq":   "Nasdaq 100 index (QQQ)",
    }
    config["display_name"] = _labels.get(config["selection_method"], scenario[:60])
    return config


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def interpret(scenario: str, initial_capital: float = 10_000, start_year: int = 2001) -> dict:
    """
    Interpret a natural-language scenario into a backtest config.
    Priority: OpenRouter → Groq → Anthropic → rule-based
    """
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

    # Always enforce UI settings
    config.setdefault("selection_method", "top_n_prior_return")
    config.setdefault("n", 1)
    config.setdefault("tickers", [])
    config.setdefault("rebalance_frequency", "annual")
    config.setdefault("selection_lookback_months", 12)
    config.setdefault("buy_offset_months", 0)
    config.setdefault("end_year", 2026)
    config.setdefault("display_name", scenario[:80])
    config["initial_capital"] = initial_capital
    config["start_year"] = start_year

    return config
