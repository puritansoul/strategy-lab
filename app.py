"""
Strategy Lab — Natural-language S&P 500 backtester
Run: streamlit run app.py
"""

import json
import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# Inject Streamlit Cloud secrets into environment variables
for _key in ("OPENROUTER_API_KEY", "GROQ_API_KEY", "ANTHROPIC_API_KEY"):
    if _key not in os.environ:
        try:
            os.environ[_key] = st.secrets[_key]
        except Exception:
            pass

from engine import DataManager, run_backtest
from interpreter import interpret

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Strategy Lab",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("📈 Strategy Lab")
st.caption("Describe any investment scenario in plain English — see 25 years of results instantly.")

# ---------------------------------------------------------------------------
# Sidebar settings
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("Settings")
    initial_capital = st.number_input("Starting capital ($)", value=10_000, step=1_000, min_value=1_000)
    start_year = st.slider("Start year", min_value=2001, max_value=2024, value=2001)
    st.divider()

    import os as _os
    _has_or = bool(_os.environ.get("OPENROUTER_API_KEY"))
    _has_groq = bool(_os.environ.get("GROQ_API_KEY"))
    _has_claude = bool(_os.environ.get("ANTHROPIC_API_KEY"))
    if _has_or or _has_groq or _has_claude:
        keys_active = []
        if _has_or: keys_active.append("OpenRouter")
        if _has_groq: keys_active.append("Groq")
        if _has_claude: keys_active.append("Claude")
        st.success(f"🤖 AI interpreter active ({', '.join(keys_active)})")
    else:
        st.warning("🔤 Rule-based interpreter only")
        with st.expander("Enable free AI interpreter"):
            st.markdown(
                "**Groq (recommended — fast & free)**\n\n"
                "1. Go to **console.groq.com** → Sign up free\n"
                "2. API Keys → Create Key\n"
                "3. Add to Streamlit secrets:\n"
                "```\nGROQ_API_KEY = 'gsk_...'\n```\n\n"
                "**OpenRouter (also free)**\n\n"
                "1. Go to **openrouter.ai** → Sign up\n"
                "2. Keys → Create Key\n"
                "3. Add to Streamlit secrets:\n"
                "```\nOPENROUTER_API_KEY = 'sk-or-...'\n```"
            )
    st.divider()
    st.caption("Data: S&P 500 composition via fja05680/sp500 · Prices via yfinance")

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def get_data_manager():
    return DataManager()

dm = get_data_manager()

if not dm.prices_loaded():
    with st.status("Loading historical price data (one-time setup, ~5 min)…", expanded=True) as status:
        progress_bar = st.progress(0)
        status_text = st.empty()

        def progress_fn(pct, msg):
            progress_bar.progress(pct)
            status_text.text(msg)

        dm.load_prices(progress_fn)
        status.update(label="Price data ready!", state="complete")
else:
    dm.load_prices()  # loads from cache silently

# ---------------------------------------------------------------------------
# Example scenarios
# ---------------------------------------------------------------------------

EXAMPLES = [
    "Buy last year's best performing stock",
    "Buy top 5 performers from prior year",
    "Buy top 5 January winners on Feb 1 each year",
    "Buy the 10 worst performers from prior year (contrarian)",
    "Buy the 3 largest companies by market cap",
    "Buy top 10 momentum stocks (6-month lookback)",
    "Just hold SPY (S&P 500 index)",
    "What if I just held NVDA since 2010?",
    "What if I held AMZN since 2002?",
]

# Persist selected scenario and auto-run flag across rerenders
if "scenario_text" not in st.session_state:
    st.session_state["scenario_text"] = ""
if "auto_run" not in st.session_state:
    st.session_state["auto_run"] = False

st.subheader("Scenario")
cols = st.columns(4)
for i, ex in enumerate(EXAMPLES):
    if cols[i % 4].button(ex, width="stretch", key=f"ex_{i}"):
        st.session_state["scenario_text"] = ex
        st.session_state["auto_run"] = True
        # No st.rerun() — button click already triggers a rerender

scenario_input = st.text_area(
    "Or describe your own scenario:",
    placeholder="e.g. 'What if I bought the top 3 performing stocks from the prior year and held equally?'",
    height=80,
    key="scenario_text",
)

run_col, _ = st.columns([1, 5])
manual_run = run_col.button("▶  Run Backtest", type="primary", width="stretch")

# Consume auto_run flag — read then immediately clear so it only fires once
auto_run = st.session_state.get("auto_run", False)
if auto_run:
    st.session_state["auto_run"] = False
run_clicked = manual_run or auto_run

# ---------------------------------------------------------------------------
# Backtest execution
# ---------------------------------------------------------------------------

def _render_strategy_explanation(config: dict, result: dict, dm) -> None:
    import pandas as pd

    method = config.get("selection_method", "top_n_prior_return")
    n = config.get("n", 1)
    freq = config.get("rebalance_frequency", "annual")
    lb = config.get("selection_lookback_months", 12)
    offset = config.get("buy_offset_months", 0)
    tickers_cfg = config.get("tickers", [])

    _method_labels = {
        "top_n_prior_return":    f"top {n} stock{'s' if n>1 else ''} by return over the lookback window",
        "bottom_n_prior_return": f"bottom {n} stock{'s' if n>1 else ''} by return (contrarian — worst performers)",
        "top_n_momentum":        f"top {n} stock{'s' if n>1 else ''} by momentum",
        "bottom_n_momentum":     f"bottom {n} stock{'s' if n>1 else ''} by momentum",
        "top_n_market_cap":      f"top {n} stock{'s' if n>1 else ''} by market cap (largest companies)",
        "hold_ticker":           f"always hold {', '.join(tickers_cfg)} — no selection needed",
        "equal_weight_sp500":    "SPY (S&P 500 index fund) — no selection needed",
        "equal_weight_nasdaq":   "QQQ (Nasdaq 100 index fund) — no selection needed",
    }
    _freq_labels = {"annual": "once per year", "quarterly": "every quarter", "monthly": "every month"}

    # Buy/sell timing — derive from actual first period data
    rows_for_desc = result.get("strategy_rows", [])
    if rows_for_desc:
        _f = rows_for_desc[0]
        _bd = _f.get("buy_date") or (pd.Timestamp(f"{_f['year']}-01-01") + pd.DateOffset(months=offset))
        _sd = _f.get("sell_date") or (pd.Timestamp(f"{_f['year']+1}-01-01") + pd.DateOffset(months=offset) - pd.Timedelta(days=1))
        _hold_months = int(round((_sd - _bd).days / 30))
        buy_timing = f"around {_bd.strftime('%B %d')} each year"
        sell_timing = f"around {_sd.strftime('%B %d')} next year (hold ~{_hold_months} months)"
    else:
        buy_timing = "start of each period"
        sell_timing = "end of each period"

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Strategy parameters**")
        st.markdown(f"""
| Field | Value |
|---|---|
| Universe | S&P 500 members at time of purchase |
| Select | {_method_labels.get(method, method)} |
| Lookback | {lb} month{'s' if lb != 1 else ''} before buy date |
| Buy | {buy_timing} |
| Sell | {sell_timing} |
| Rebalance | {_freq_labels.get(freq, freq)} |
| Position sizing | Equal weight across all {n} position{'s' if n>1 else ''} |
""")

    with col2:
        rows = result.get("strategy_rows", [])
        if not rows:
            st.info("No example data available.")
            return

        first = rows[0]
        year = first["year"]
        buy_date = first.get("buy_date") or (pd.Timestamp(f"{year}-01-01") + pd.DateOffset(months=offset))
        sell_date = first.get("sell_date") or pd.Timestamp(f"{year}-12-31")
        sel_start = buy_date - pd.DateOffset(months=lb)
        universe = dm.members_at_date(buy_date)

        st.markdown(f"**Concrete example: {year}**")

        st.markdown(
            f"**Universe:** ~{len(universe)} S&P 500 stocks on "
            f"{buy_date.strftime('%b %d, %Y')}"
        )
        if method not in ("hold_ticker", "equal_weight_sp500", "equal_weight_nasdaq"):
            st.markdown(
                f"**Lookback window:** {sel_start.strftime('%b %d, %Y')} → "
                f"{buy_date.strftime('%b %d, %Y')} ({lb} month{'s' if lb!=1 else ''})"
            )

        st.markdown(
            f"**Buy:** {buy_date.strftime('%B %d, %Y')}  "
            f"**Sell:** {sell_date.strftime('%B %d, %Y')}"
        )

        tickers = first.get("tickers", [])
        hold_rets = first.get("ticker_returns", {})
        if tickers:
            # Fetch selection-period returns for display
            sel_data = []
            for tk in tickers:
                sel_r = dm.return_between(tk, str(sel_start.date()), str(buy_date.date()))
                hold_r = hold_rets.get(tk)
                sel_data.append({
                    "Ticker": tk,
                    f"Return {sel_start.strftime('%b')}–{buy_date.strftime('%b')} (why selected)": (
                        f"{sel_r*100:+.1f}%" if sel_r is not None else "n/a"
                    ),
                    f"Return held {buy_date.strftime('%b')}–{sell_date.strftime('%b')}": (
                        f"{hold_r*100:+.1f}%" if hold_r is not None else "n/a"
                    ),
                })
            st.dataframe(pd.DataFrame(sel_data), hide_index=True, width="stretch")
            st.markdown(
                f"**Portfolio return {year}:** "
                f"`{first['portfolio_return']*100:+.1f}%`  "
                f"(equal-weight average of {len(tickers)} stocks)"
            )


if run_clicked and scenario_input.strip():
    with st.spinner("Interpreting scenario…"):
        try:
            config = interpret(scenario_input.strip(), initial_capital, start_year)
        except Exception as e:
            st.error(f"Could not interpret scenario: {e}")
            st.stop()
        if config.get("_interpreter") == "rule-based":
            st.caption("🔤 Interpreted via keyword matching — verify the strategy explanation below.")

    # If market cap strategy and shares not cached yet, download now
    if config.get("selection_method") == "top_n_market_cap" and not dm.shares_loaded():
        with st.status("Downloading shares outstanding data (one-time, ~60s)…", expanded=True) as st_shares:
            dm.shares_cache()
            st_shares.update(label="Shares data ready!", state="complete")

    with st.spinner("Running backtest…"):
        result = run_backtest(config, dm)

    # -- Strategy explanation --
    st.divider()
    st.subheader(f"Results: _{config.get('display_name', scenario_input[:60])}_")

    with st.expander("📋 What the AI understood — strategy explanation + real example", expanded=True):
        _render_strategy_explanation(config, result, dm)

    # -- Summary cards --
    init = result["initial_capital"]
    final = result["final_value"]
    spy_final = result["spy_final"]
    total_roi = result["total_roi"]
    spy_roi = result["spy_roi"]

    gain_vs_spy = final - spy_final

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Final Portfolio", f"${final:,.0f}", f"{total_roi*100:+.1f}%")
    c2.metric("S&P 500 (same period)", f"${spy_final:,.0f}", f"{spy_roi*100:+.1f}%")
    c3.metric("vs S&P 500", f"${gain_vs_spy:+,.0f}", f"{(total_roi - spy_roi)*100:+.1f}pp")
    c4.metric("Starting capital", f"${init:,.0f}", f"{start_year}–2026")

    # -- Chart --
    rows = result["strategy_rows"]
    spy_rows = result["spy_rows"]

    years = [r["year"] for r in rows]
    values = [r["end_value"] for r in rows]
    spy_values = [r["value"] for r in spy_rows if r["year"] in years]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=years, y=values,
        mode="lines+markers",
        name="Your Strategy",
        line=dict(color="#00e676", width=2.5),
        marker=dict(size=6),
        hovertemplate="Year: %{x}<br>Portfolio: $%{y:,.0f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=[r["year"] for r in spy_rows],
        y=[r["value"] for r in spy_rows],
        mode="lines",
        name="S&P 500 (SPY)",
        line=dict(color="#2196f3", width=2, dash="dot"),
        hovertemplate="Year: %{x}<br>SPY: $%{y:,.0f}<extra></extra>",
    ))
    fig.update_layout(
        title="Portfolio Value Over Time",
        xaxis_title="Year",
        yaxis_title="Portfolio Value ($)",
        template="plotly_dark",
        hovermode="x unified",
        height=420,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        yaxis=dict(tickformat="$,.0f"),
    )
    st.plotly_chart(fig, width="stretch")

    # -- Year-by-year table --
    st.subheader("Year-by-Year Breakdown")

    spy_by_year = {r["year"]: r["return"] for r in spy_rows}

    def _color(val_str):
        try:
            num = float(val_str.replace("%", "").replace("+", "").replace("pp", ""))
            return "#1a7a3a" if num > 0 else "#c0392b" if num < 0 else "inherit"
        except Exception:
            return "inherit"

    def _held_cell(tickers):
        if not tickers:
            return "—"
        if len(tickers) <= 5:
            return ", ".join(tickers)
        preview = ", ".join(tickers[:5])
        rest = ", ".join(tickers[5:])
        return (
            f"{preview}, "
            f"<details style='display:inline;'>"
            f"<summary style='display:inline;cursor:pointer;color:#1565c0;'>"
            f"(+{len(tickers)-5} more)</summary>"
            f"<span> {rest}</span>"
            f"</details>"
        )

    spy_value = float(config.get("initial_capital", 10_000.0))
    spy_values_by_year = {}
    for sr in spy_rows:
        spy_values_by_year[sr["year"]] = sr["value"]

    html_rows = ""
    for r in rows:
        held_cell = _held_cell(r["tickers"])

        if r["ticker_returns"]:
            best_tk = max(r["ticker_returns"], key=lambda k: r["ticker_returns"][k])
            best_ret = r["ticker_returns"][best_tk]
            best_str = f"{best_tk} {best_ret*100:+.1f}%"
            best_color = _color(f"{best_ret*100:+.1f}%")
        else:
            best_str, best_color = "—", "#aaa"

        ret_pct = r["portfolio_return"] * 100
        spy_pct = spy_by_year.get(r["year"], 0) * 100
        delta = ret_pct - spy_pct
        spy_val = spy_values_by_year.get(r["year"])
        spy_val_str = f"${spy_val:,.0f}" if spy_val is not None else "—"

        html_rows += f"""
        <tr>
          <td style='padding:6px 12px;'>{r['year']}</td>
          <td style='padding:6px 12px;font-size:0.88em;'>{held_cell}</td>
          <td style='padding:6px 12px;color:{best_color};font-size:0.88em;'>{best_str}</td>
          <td style='padding:6px 12px;color:{_color(f"{ret_pct:+.1f}%")};text-align:right;font-weight:600;'>{ret_pct:+.1f}%</td>
          <td style='padding:6px 12px;color:{_color(f"{spy_pct:+.1f}%")};text-align:right;font-weight:600;'>{spy_pct:+.1f}%</td>
          <td style='padding:6px 12px;color:{_color(f"{delta:+.1f}pp")};text-align:right;font-weight:600;'>{delta:+.1f}pp</td>
          <td style='padding:6px 12px;text-align:right;font-weight:600;color:{"#1a7a3a" if spy_val is None or r["end_value"] >= spy_val else "#c0392b"};'>${r['end_value']:,.0f}</td>
          <td style='padding:6px 12px;text-align:right;'>{spy_val_str}</td>
        </tr>"""

    html_table = f"""
    <style>
      .strat-table {{ border-collapse: collapse; width: 100%; font-size:0.93em; }}
      .strat-table th {{ font-weight:600; padding:8px 12px;
                         border-bottom:2px solid #ddd; text-align:left; }}
      .strat-table td {{ border-bottom:1px solid #eee; }}
      .strat-table tr:hover td {{ background:rgba(0,0,0,0.03); }}
      details summary::-webkit-details-marker {{ display:none; }}
    </style>
    <table class='strat-table'>
      <thead><tr>
        <th>Year</th>
        <th>Held</th>
        <th>Best ticker</th>
        <th style='text-align:right;'>Strategy return</th>
        <th style='text-align:right;'>SPY return</th>
        <th style='text-align:right;'>vs SPY</th>
        <th style='text-align:right;'>Portfolio value</th>
        <th style='text-align:right;'>SPY value</th>
      </tr></thead>
      <tbody>{html_rows}</tbody>
    </table>
    """
    st.markdown(html_table, unsafe_allow_html=True)

    # -- Strategy config debug (expander) --
    with st.expander("Strategy config (what the AI interpreted)"):
        st.json({k: v for k, v in config.items() if not k.startswith("_")})

elif run_clicked:
    st.warning("Please enter a scenario first.")
