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
- "top_n_prior_return"    : buy the top N stocks by prior year's return (chasing winners / momentum)
- "bottom_n_prior_return" : buy the bottom N stocks by prior year's return (contrarian / mean reversion)
- "top_n_momentum"        : buy top N stocks by recent price momentum (shorter lookback)
- "bottom_n_momentum"     : buy worst N stocks by recent price momentum
- "top_n_market_cap"      : buy top N stocks by market cap (largest companies)
- "hold_ticker"           : simply hold specific named ticker(s) every year
- "equal_weight_sp500"    : hold SPY (S&P 500 index fund)
- "equal_weight_nasdaq"   : hold QQQ (Nasdaq 100 index fund)

PARAMETERS:
- n: number of stocks to hold (1 = concentrated single stock, 5/10/20 = diversified)
- tickers: list of ticker symbols — ONLY for hold_ticker method
- lookback_months: lookback in months for momentum (3, 6, or 12). Default 12.
- start_year: first year to trade (2001–2024). Default 2001.
- end_year: last year (use 2026 for current). Default 2026.
- initial_capital: starting $ (default 10000)
- weight: always "equal"

RULES:
- "best", "top", "highest return", "winners", "best performing" → top_n_prior_return
- "worst", "bottom", "losers", "contrarian", "underperformers" → bottom_n_prior_return
- "biggest", "largest", "market cap", "most valuable" → top_n_market_cap
- "momentum", "trending", "hot", "breakout" → top_n_momentum
- "hold X", "buy X", "invest in X", "just X" (specific company/ticker) → hold_ticker
- "S&P 500 index", "just SPY", "index fund" → equal_weight_sp500
- If user says "top 5" or "5 stocks" or "best 5" or "5 best" → n=5
- If no number mentioned → n=1
- If start year not mentioned → use 2001

Respond ONLY with raw JSON. No explanation, no markdown fences.

Example output:
{"selection_method":"top_n_prior_return","n":5,"tickers":[],"lookback_months":12,"start_year":2001,"end_year":2026,"initial_capital":10000,"weight":"equal","display_name":"Top 5 prior-year winners (annual)"}
"""


def _call_gemini(scenario: str, initial_capital: float, start_year: int) -> dict:
    from google import genai
    from google.genai import types
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    user_msg = f"Scenario: {scenario}\nInitial capital: ${initial_capital:,.0f}\nStart year: {start_year}"
    resp = client.models.generate_content(
        model="gemini-flash-lite-latest",
        contents=user_msg,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            temperature=0,
            max_output_tokens=300,
        ),
    )
    raw = resp.text.strip()
    raw = re.sub(r"```(?:json)?", "", raw).strip("`").strip()
    return json.loads(raw)


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
        "lookback_months": 12,
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

    # ---- selection method ----
    if re.search(r'\b(hold|held|keep|just buy|just hold|bought|invest in|put.*into)\b', text) and tickers_found:
        config["selection_method"] = "hold_ticker"
        config["tickers"] = tickers_found
        config["n"] = len(tickers_found)
    elif "SPY" in scenario or re.search(r'\b(s&p\s*500\s*index|just.*index|index fund)\b', text):
        config["selection_method"] = "equal_weight_sp500"
    elif "QQQ" in scenario or re.search(r'\b(nasdaq\s*100|qqq)\b', text):
        config["selection_method"] = "equal_weight_nasdaq"
    elif re.search(r'\b(worst|bottom|loser|losing|losers|loss|losses|contrarian|fallen|fell|dropped|declined|underperform|mean.?revert|bad|poor|laggard|laggards|dog|dogs)\b', text):
        config["selection_method"] = "bottom_n_prior_return"
    elif re.search(r'\b(market.?cap|largest|biggest|most valuable|mega.?cap)\b', text):
        config["selection_method"] = "top_n_market_cap"
    elif re.search(r'\b(momentum|trending|hot|breakout|relative strength)\b', text):
        config["selection_method"] = "top_n_momentum"
        if re.search(r'\b(3.?month|quarter|90.?day)\b', text):
            config["lookback_months"] = 3
        elif re.search(r'\b(6.?month|half.?year|180.?day)\b', text):
            config["lookback_months"] = 6
    else:
        config["selection_method"] = "top_n_prior_return"

    # ---- detect start year ----
    m = re.search(r'\b(20\d\d)\b', text)
    if m:
        yr = int(m.group(1))
        if 2001 <= yr <= 2025:
            config["start_year"] = yr

    # ---- display name ----
    _labels = {
        "top_n_prior_return":    f"Top {n} prior-year winner{'s' if n>1 else ''} (annual)",
        "bottom_n_prior_return": f"Bottom {n} prior-year loser{'s' if n>1 else ''} (contrarian)",
        "top_n_momentum":        f"Top {n} momentum stock{'s' if n>1 else ''} ({config['lookback_months']}m lookback)",
        "bottom_n_momentum":     f"Bottom {n} momentum stock{'s' if n>1 else ''}",
        "top_n_market_cap":      f"Top {n} largest by market cap",
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
    Tries Groq → Anthropic → rule-based, in that order.
    """
    config = None
    error_msg = None

    if os.environ.get("GEMINI_API_KEY"):
        try:
            config = _call_gemini(scenario, initial_capital, start_year)
        except Exception as e:
            error_msg = f"Gemini: {e}"

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
    else:
        config["_interpreter"] = "llm"

    # Always enforce UI settings
    config.setdefault("selection_method", "top_n_prior_return")
    config.setdefault("n", 1)
    config.setdefault("tickers", [])
    config.setdefault("lookback_months", 12)
    config.setdefault("end_year", 2026)
    config.setdefault("display_name", scenario[:80])
    config["initial_capital"] = initial_capital
    config["start_year"] = start_year

    return config
