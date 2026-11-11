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
- "top_n_market_cap_growth": buy top N stocks by % increase in market cap over lookback
- "hold_ticker"           : simply hold specific named ticker(s) every period
- "equal_weight_sp500"    : hold SPY (S&P 500 index fund)
- "equal_weight_nasdaq"   : hold QQQ (Nasdaq 100 index fund)

PARAMETERS:
- n: number of stocks to hold (1 = concentrated, 5/10/20 = diversified)
- tickers: list of ticker symbols — ONLY for hold_ticker method
- rebalance_frequency: "annual" (default), "quarterly", or "monthly" — how often to re-pick stocks
- selection_lookback_months: months to look back for ranking (default: 12 annual, 3 quarterly, 1 monthly)
- buy_offset_months: months to delay buying after period start (default 0; use 1 for Jan effect, 3 to buy in April)
- hold_months: how many months to hold each position before selling (default: null = full period length). Set explicitly when user specifies a different hold duration, e.g. "hold 9 months", "sell in December", "hold until next April" (=12 when buy is April)
- start_year: first year to trade (2001-2024). Default 2001.
- end_year: last year (use 2026 for current). Default 2026.
- initial_capital: starting $ (default 10000)
- weight: always "equal"
- filters: SQL-style WHERE conditions applied to narrow the universe BEFORE ranking. List of {"field", "op", "value"}.
  Supported fields: "market_cap" (price × shares), "price" (stock price at buy date), "prior_return" (return over lookback)
  Supported ops: ">", "<", ">=", "<=", "=", "!="
  Examples:
    "market cap > $10B"   → {"field":"market_cap","op":">","value":10000000000}
    "large cap"           → {"field":"market_cap","op":">","value":10000000000}
    "mega cap"            → {"field":"market_cap","op":">","value":100000000000}
    "mid cap"             → [{"field":"market_cap","op":">","value":2000000000},{"field":"market_cap","op":"<","value":10000000000}]
    "small cap"           → {"field":"market_cap","op":"<","value":2000000000}
    "price above $50"     → {"field":"price","op":">","value":50}
    "price below $20"     → {"field":"price","op":"<","value":20}
  If no filter mentioned → filters: []

RULES:
- "best", "top", "winners", "best performing" → top_n_prior_return
- "worst", "bottom", "losers", "contrarian" → bottom_n_prior_return
- "market cap", "market caps", "largest", "biggest", "most valuable" → top_n_market_cap
- "market cap growth", "market cap increased", "grew most in market cap", "biggest market cap gain" → top_n_market_cap_growth
- "momentum", "trending", "hot", "breakout" → top_n_momentum
- "hold X", "buy X", "invest in X", "just X" (specific company) → hold_ticker
- "S&P 500 index", "just SPY", "index fund" → equal_weight_sp500
- "January effect", "best in January", "buy in February based on January" → top_n_prior_return + buy_offset_months=1 + selection_lookback_months=1
- "first quarter", "Q1 performers", "buy in April based on Q1" → top_n_prior_return + buy_offset_months=3 + selection_lookback_months=3 + rebalance_frequency=annual
- "second quarter", "Q2 performers" → top_n_prior_return + buy_offset_months=6 + selection_lookback_months=3 + rebalance_frequency=annual
- "every quarter", "quarterly rotation", "rebalance quarterly" → rebalance_frequency=quarterly, selection_lookback_months=3
- "every month", "monthly rotation", "rebalance monthly" → rebalance_frequency=monthly, selection_lookback_months=1
- If user says "top 5" or "5 stocks" or "best 5" → n=5
- If user says "all", "every", "each" with no number → n=999 (take all qualifying stocks)
  - "all losers" → n=999 AND add filter {"field":"prior_return","op":"<","value":0}
  - "all winners" → n=999 AND add filter {"field":"prior_return","op":">","value":0}
- If no number mentioned → n=1
- If start year not mentioned → use 2001
- NOTE: "hold for 3 quarters" with "rebalance every year" means buy_offset_months=3 (buy in April after Q1 ends), NOT quarterly rebalancing

IMPORTANT: Your entire response must be a single valid JSON object. No explanation, no markdown, no code fences, no extra text — just the raw JSON object starting with { and ending with }.

Example — "buy top 5 Q1 performers, hold 3 quarters, rebalance annually":
{"selection_method":"top_n_prior_return","n":5,"tickers":[],"rebalance_frequency":"annual","selection_lookback_months":3,"buy_offset_months":3,"hold_months":null,"filters":[],"start_year":2001,"end_year":2026,"initial_capital":10000,"weight":"equal","display_name":"Top 5 Q1 winners, hold Apr–Dec (annual)"}

Example — "buy the 5 worst performers with market cap > $10B":
{"selection_method":"bottom_n_prior_return","n":5,"tickers":[],"rebalance_frequency":"annual","selection_lookback_months":12,"buy_offset_months":0,"hold_months":null,"filters":[{"field":"market_cap","op":">","value":10000000000}],"start_year":2001,"end_year":2026,"initial_capital":10000,"weight":"equal","display_name":"Bottom 5 prior-year losers, mktcap > $10B (annual)"}

Example — default annual (no filters):
{"selection_method":"top_n_prior_return","n":5,"tickers":[],"rebalance_frequency":"annual","selection_lookback_months":12,"buy_offset_months":0,"hold_months":null,"filters":[],"start_year":2001,"end_year":2026,"initial_capital":10000,"weight":"equal","display_name":"Top 5 prior-year winners (annual)"}
"""


def _extract_json(text: str) -> str:
    """Extract the first valid JSON object from text that may contain reasoning or thinking tags."""
    # Strip thinking/reasoning tags (Nemotron, DeepSeek-R1, etc.)
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"<reasoning>.*?</reasoning>", "", text, flags=re.DOTALL)
    # Strip markdown fences
    text = re.sub(r"```(?:json)?", "", text).strip("`").strip()
    # Find the outermost { ... } block
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


_openrouter_free_models_cache = []
_model_empty_strikes = {}  # model_id -> consecutive empty-content failures


def _parse_params_b(model_id: str) -> float:
    """Extract parameter count in billions from a model ID string."""
    import re as _re
    m = _re.search(r'(\d+(?:\.\d+)?)\s*b(?:\b|_|-)', model_id.lower())
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
            # skip tiny models (< 3B) — too unreliable for structured JSON output
            capable = [m for m in free if _parse_params_b(m["id"]) >= 3 or _parse_params_b(m["id"]) == 0]
            # rank by parameter count descending; unknown size sorted last
            ranked = sorted(capable, key=lambda m: _parse_params_b(m["id"]), reverse=True)
            _openrouter_free_models_cache = [m["id"] for m in ranked[:10]]
            return _openrouter_free_models_cache
    except Exception:
        pass
    return []


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
    all_models = _get_free_models(key)
    if not all_models:
        raise Exception("No free models available on OpenRouter right now — try again shortly.")
    # Skip models with 2+ consecutive empty-content failures this session
    models = [m for m in all_models if _model_empty_strikes.get(m, 0) < 2]
    if not models:
        models = all_models  # reset if everything is struck out
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
                _model_empty_strikes[model] = _model_empty_strikes.get(model, 0) + 1
                errors.append(f"{model}: empty content")
                continue
            raw = _extract_json(raw)
            if not raw:
                _model_empty_strikes[model] = _model_empty_strikes.get(model, 0) + 1
                errors.append(f"{model}: no JSON found in response")
                continue
            _model_empty_strikes[model] = 0  # reset on success
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
    # "all" → unlimited (999 = take every qualifying stock after filters)
    _all_match = re.search(r'\b(all|every|each)\b', text) and not re.search(r'\b\d+\b', text)
    n = 999 if _all_match else 1
    if n == 1:
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
    elif re.search(r'market.?cap.*(grew|grow|increas|gain|rose|jump|climb|most|biggest|highest).*(year|prior|previous|last)|'
                   r'(grew|grow|increas|gain|rose|jump|climb).*(market.?cap)', text):
        config["selection_method"] = "top_n_market_cap_growth"
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

    # ---- filters (WHERE conditions) ----
    filters = []

    # Market cap: explicit dollar amount, e.g. "market cap > $10B", "over 5 billion"
    _mcap_pat = re.search(
        r'market.?cap\s*(?:above|over|greater than|>|more than|exceeding|of at least|>=)?\s*\$?\s*(\d+(?:\.\d+)?)\s*(b(?:illion)?|t(?:rillion)?|m(?:illion)?)\b',
        text
    )
    if _mcap_pat:
        _num = float(_mcap_pat.group(1))
        _unit = _mcap_pat.group(2)[0]
        _mult = {"b": 1e9, "t": 1e12, "m": 1e6}[_unit]
        filters.append({"field": "market_cap", "op": ">", "value": _num * _mult})
    elif re.search(r'\bmega.?cap\b', text):
        filters.append({"field": "market_cap", "op": ">", "value": 100_000_000_000})
    elif re.search(r'\blarge.?cap\b', text):
        filters.append({"field": "market_cap", "op": ">", "value": 10_000_000_000})
    elif re.search(r'\bmid.?cap\b', text):
        filters.append({"field": "market_cap", "op": ">", "value": 2_000_000_000})
        filters.append({"field": "market_cap", "op": "<", "value": 10_000_000_000})
    elif re.search(r'\bsmall.?cap\b', text):
        filters.append({"field": "market_cap", "op": "<", "value": 2_000_000_000})

    # Price filter: "price above $50", "stock price > 100", "price below $20"
    _price_pat = re.search(
        r'(?:stock\s+)?price\s*(above|over|>|greater than|more than|below|under|<|less than)\s*\$?\s*(\d+(?:\.\d+)?)',
        text
    )
    if _price_pat:
        _op_word = _price_pat.group(1)
        _op = ">" if _op_word in ("above", "over", ">", "greater than", "more than") else "<"
        filters.append({"field": "price", "op": _op, "value": float(_price_pat.group(2))})

    # When "all losers/winners" is used (n=999), add implicit return direction filter
    # so only stocks that actually lost/gained are included — not just ranked last/first.
    if n == 999:
        has_return_filter = any(f.get("field") == "prior_return" for f in filters)
        if not has_return_filter:
            if re.search(r'\b(worst|loser|losers|lost|loss|declined|fell|dropped|down|negative)\b', text):
                filters.append({"field": "prior_return", "op": "<", "value": 0})
            elif re.search(r'\b(best|winner|winners|gained|rose|up|positive|returned)\b', text):
                filters.append({"field": "prior_return", "op": ">", "value": 0})

    config["filters"] = filters

    # ---- hold_months ----
    hm = re.search(r'\bhold(?:ing)?\s+(?:for\s+)?(\d+)\s*(month|quarter)', text)
    if hm:
        num = int(hm.group(1))
        unit = hm.group(2)
        config["hold_months"] = num * 3 if unit.startswith("quarter") else num
    elif re.search(r'\bhold(?:ing)?\s+(?:for\s+)?(?:a\s+)?(?:full\s+)?year\b', text):
        config["hold_months"] = 12
    elif re.search(r'\bhold(?:ing)?\s+(?:for\s+)?(?:a\s+)?(?:full\s+)?quarter\b', text):
        config["hold_months"] = 3

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
        "top_n_prior_return":    (f"Top {n} January winners (buy Feb, hold 12mo)" if jan_effect
                                  else f"Top {n} Q1 winners (buy Apr, hold 12mo)" if q1_effect
                                  else f"Top {n} prior-{lb}mo winner{'s' if n>1 else ''} ({freq_label})"),
        "bottom_n_prior_return": f"Bottom {n} prior-{lb}mo loser{'s' if n>1 else ''} ({freq_label})",
        "top_n_momentum":        f"Top {n} momentum stock{'s' if n>1 else ''} ({lb}m, {freq_label})",
        "bottom_n_momentum":     f"Bottom {n} momentum stock{'s' if n>1 else ''} ({lb}m, {freq_label})",
        "top_n_market_cap":          f"Top {n} largest by market cap ({freq_label})",
        "top_n_market_cap_growth":   f"Top {n} by market cap growth ({lb}mo, {freq_label})",
        "hold_ticker":           f"Hold {', '.join(config['tickers'])}",
        "equal_weight_sp500":    "S&P 500 index (SPY)",
        "equal_weight_nasdaq":   "Nasdaq 100 index (QQQ)",
    }
    _filter_suffix = ""
    for _f in filters:
        _fld = _f.get("field", "")
        _fop = _f.get("op", ">")
        _fval = float(_f.get("value", 0))
        if _fld == "market_cap":
            if _fval >= 1e12:
                _fstr = f"${_fval/1e12:.1f}T"
            elif _fval >= 1e9:
                _fstr = f"${_fval/1e9:.0f}B"
            else:
                _fstr = f"${_fval/1e6:.0f}M"
            _filter_suffix += f", mktcap {_fop} {_fstr}"
        elif _fld == "price":
            _filter_suffix += f", price {_fop} ${_fval:.0f}"

    config["display_name"] = _labels.get(config["selection_method"], scenario[:60]) + _filter_suffix
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
    config.setdefault("filters", [])
    config.setdefault("end_year", 2026)
    config.setdefault("display_name", scenario[:80])
    config["initial_capital"] = initial_capital
    config["start_year"] = start_year

    return config
