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

# ---------------------------------------------------------------------------
# Compact horizontal settings bar (top of page, before title)
# ---------------------------------------------------------------------------

import os as _os
_has_or = bool(_os.environ.get("OPENROUTER_API_KEY"))
_has_groq = bool(_os.environ.get("GROQ_API_KEY"))
_has_claude = bool(_os.environ.get("ANTHROPIC_API_KEY"))
_ai_active = _has_or or _has_groq or _has_claude

_tb1, _tb2, _tb3, _tb4 = st.columns([1.4, 1.6, 1.5, 2.5])
with _tb1:
    st.caption("Starting capital ($)")
    initial_capital = st.number_input(
        "Starting capital ($)", value=10_000, step=1_000, min_value=1_000,
        label_visibility="collapsed",
    )
with _tb2:
    st.caption(f"Start year")
    start_year = st.slider(
        "Start year", min_value=2001, max_value=2024, value=2001,
        label_visibility="collapsed",
    )
with _tb3:
    st.caption("AI interpreter")
    if _ai_active:
        keys_active = []
        if _has_or: keys_active.append("OpenRouter")
        if _has_groq: keys_active.append("Groq")
        if _has_claude: keys_active.append("Claude")
        st.markdown(f"🟢 **{', '.join(keys_active)}**")
    else:
        st.markdown("🟡 Rule-based only")
with _tb4:
    st.caption("Data sources")
    st.caption("S&P 500 composition: fja05680/sp500 · Prices: yfinance")

st.divider()

# ---------------------------------------------------------------------------
# Title
# ---------------------------------------------------------------------------

st.title("📈 Strategy Lab")
st.caption("Describe any investment scenario in plain English — see 25 years of results instantly.")

# ---------------------------------------------------------------------------
# Sidebar: saved backtests only
# ---------------------------------------------------------------------------

if "saved_backtests" not in st.session_state:
    st.session_state["saved_backtests"] = []

with st.sidebar:

    # Saved backtests compact list
    _saved = st.session_state.get("saved_backtests", [])
    if _saved:
        st.subheader(f"Saved Backtests ({len(_saved)})")
        _to_delete = None
        for _i, _s in enumerate(_saved):
            _beat_pct = f"{_s['years_beat']}/{_s['total_years']} yrs beat S&P"
            with st.container(border=True):
                _nc, _dc = st.columns([5, 1])
                _nc.markdown(f"**{_s['name'][:35]}**")
                _nc.caption(f"${_s['final']:,.0f} · {_s['total_roi']*100:+.1f}% · {_beat_pct}")
                if _dc.button("✕", key=f"sidebar_del_{_i}", help="Remove"):
                    _to_delete = _i
        if _to_delete is not None:
            st.session_state["saved_backtests"].pop(_to_delete)
            st.rerun()
        if len(_saved) > 1:
            if st.button("🗑 Clear all", key="sidebar_clear_all"):
                st.session_state["saved_backtests"] = []
                st.rerun()

        # Compare — checkboxes when 2+ saves exist
        if len(_saved) >= 2:
            st.divider()
            st.markdown("**Compare** *(select 2 or 3)*")
            _checked_indices = []
            for _ci, _cs in enumerate(_saved):
                _already = _ci in st.session_state.get("_cmp_checked", [])
                if st.checkbox(_cs["name"][:42], value=_already, key=f"cmp_chk_{_ci}"):
                    _checked_indices.append(_ci)
            st.session_state["_cmp_checked"] = _checked_indices
            _n_checked = len(_checked_indices)
            if _n_checked >= 2:
                if st.button("Compare", type="primary", key="sidebar_compare_btn"):
                    st.session_state["_compare_indices"] = _checked_indices[:3]
                    st.rerun()
            elif _n_checked == 1:
                st.caption("Select at least one more")

        st.divider()

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

_DM_VERSION = "3"  # bump when DataManager API changes to force cache invalidation

@st.cache_resource(show_spinner=False)
def get_data_manager(version: str = _DM_VERSION):
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

# Persist selected scenario, auto-run flag, and last backtest result across rerenders
if "scenario_text" not in st.session_state:
    st.session_state["scenario_text"] = ""
if "auto_run" not in st.session_state:
    st.session_state["auto_run"] = False
if "last_run_config" not in st.session_state:
    st.session_state["last_run_config"] = None
if "last_run_result" not in st.session_state:
    st.session_state["last_run_result"] = None

with st.expander("Scenario Examples", expanded=True):
    cols = st.columns(4)
    for i, ex in enumerate(EXAMPLES):
        if cols[i % 4].button(ex, width="stretch", key=f"ex_{i}"):
            st.session_state["scenario_text"] = ex
            st.session_state["auto_run"] = True

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
        "top_n_market_cap":        f"top {n} stock{'s' if n>1 else ''} by market cap (largest companies)",
        "top_n_market_cap_growth": f"top {n} stock{'s' if n>1 else ''} by market cap growth (biggest gainers)",
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

    # Build filter description for display
    def _fmt_filter(f):
        field = f.get("field", "")
        op = f.get("op", ">")
        val = float(f.get("value", 0))
        if field == "market_cap":
            if val >= 1e12:
                val_str = f"${val/1e12:.1f}T"
            elif val >= 1e9:
                val_str = f"${val/1e9:.0f}B"
            else:
                val_str = f"${val/1e6:.0f}M"
            return f"Market cap {op} {val_str}"
        if field == "price":
            return f"Price {op} ${val:.0f}"
        if field == "prior_return":
            return f"Prior return {op} {val*100:.1f}%"
        return f"{field} {op} {val}"

    filters = config.get("filters", [])
    filter_desc = " AND ".join(_fmt_filter(f) for f in filters) if filters else "None"

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Strategy parameters**")
        st.markdown(f"""
| Field | Value |
|---|---|
| Universe | S&P 500 members at time of purchase |
| Filters | {filter_desc} |
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

    with st.spinner("Running backtest…"):
        result = run_backtest(config, dm)

    st.session_state["last_run_config"] = config
    st.session_state["last_run_result"] = result

elif run_clicked:
    st.warning("Please enter a scenario first.")

# Display results from current run or persisted previous run
_config = st.session_state.get("last_run_config")
_result = st.session_state.get("last_run_result")

if _config is not None and _result is not None:
    config = _config
    result = _result

    # -- Strategy explanation --
    st.divider()
    st.subheader(f"Results: _{config.get('display_name', '')}_")

    with st.expander("📋 What the AI understood — strategy explanation + real example", expanded=False):
        _render_strategy_explanation(config, result, dm)

    # -- Compute summary stats --
    init = result["initial_capital"]
    final = result["final_value"]
    spy_final = result["spy_final"]
    total_roi = result["total_roi"]
    spy_roi = result["spy_roi"]
    rows = result["strategy_rows"]
    spy_rows = result["spy_rows"]
    spy_by_year_ret = {r["year"]: r["return"] for r in spy_rows}
    spy_val_by_year = {r["year"]: r["value"] for r in spy_rows}

    years_beat = sum(1 for r in rows if r["portfolio_return"] > spy_by_year_ret.get(r["year"], 0))
    total_years = len(rows)
    years_under_spy_val = sum(1 for r in rows if r["end_value"] < spy_val_by_year.get(r["year"], 0))

    max_behind, cur_behind = 0, 0
    for r in rows:
        if r["portfolio_return"] < spy_by_year_ret.get(r["year"], 0):
            cur_behind += 1
            max_behind = max(max_behind, cur_behind)
        else:
            cur_behind = 0

    port_min = min((r["end_value"] for r in rows), default=init)
    spy_min  = min((r["value"]     for r in spy_rows), default=init)

    # -- Summary tiles: single row of 7 --
    def _tile(icon, label, value, sub, sub_good):
        sub_color = "#00c853" if sub_good else "#f44336"
        tint = "0,200,83" if sub_good else "244,67,54"
        return f"""<div style="flex:1;min-width:0;border-radius:12px;padding:16px 14px;
            border:1px solid rgba({tint},0.3);
            background:linear-gradient(145deg,rgba({tint},0.07),rgba({tint},0.01));">
          <div style="font-size:0.7em;text-transform:uppercase;letter-spacing:0.05em;
               color:#888;font-weight:600;margin-bottom:8px;line-height:1.35;">{icon} {label}</div>
          <div style="font-size:1.3em;font-weight:700;line-height:1.2;white-space:nowrap;">{value}</div>
          <div style="font-size:0.75em;color:{sub_color};margin-top:6px;line-height:1.35;">{sub}</div>
        </div>"""

    beat_good = years_beat >= total_years / 2
    under_good = years_under_spy_val <= total_years / 2
    streak_good = max_behind <= 3
    port_good = final > spy_final
    port_low_good = port_min >= init
    spy_low_good = spy_min >= init

    tiles = "".join([
        _tile("📈", "Final Portfolio",    f"${final:,.0f}",     f"{total_roi*100:+.1f}% total return",       port_good),
        _tile("📊", "S&amp;P 500",        f"${spy_final:,.0f}", f"{spy_roi*100:+.1f}% total return",         spy_roi > 0),
        _tile("🏆", "Years Beat S&amp;P", f"{years_beat}/{total_years}", f"{years_beat/total_years*100:.0f}% of years", beat_good),
        _tile("📉", "Yrs Under S&amp;P $", f"{years_under_spy_val}/{total_years}", f"{years_under_spy_val/total_years*100:.0f}% of years portfolio lagged", under_good),
        _tile("🔁", "Max Losing Streak", f"{max_behind} yr{'s' if max_behind!=1 else ''}", "consecutive yrs behind S&amp;P", streak_good),
        _tile("💰", "Portfolio Low",      f"${port_min:,.0f}",  f"{(port_min-init)/init*100:+.1f}% vs start", port_low_good),
        _tile("📉", "S&amp;P Low",        f"${spy_min:,.0f}",   f"{(spy_min-init)/init*100:+.1f}% vs start",  spy_low_good),
    ])
    st.markdown(
        f'<div style="display:flex;gap:8px;margin:20px 0 24px;">{tiles}</div>',
        unsafe_allow_html=True,
    )

    # -- Save button --
    save_key = f"save_{hash(config.get('display_name','') + str(final))}"
    already_saved = any(s["name"] == config.get("display_name") and s["final"] == final
                        for s in st.session_state["saved_backtests"])
    if already_saved:
        st.caption("✅ Already saved")
    elif st.button("💾 Save this backtest", key=save_key):
        st.session_state["saved_backtests"].append({
            "name":              config.get("display_name", ""),
            "final":             final,
            "spy_final":         spy_final,
            "total_roi":         total_roi,
            "spy_roi":           spy_roi,
            "years_beat":        years_beat,
            "total_years":       total_years,
            "max_behind":        max_behind,
            "years_under_spy_val": years_under_spy_val,
            "port_min":          port_min,
            "spy_min":           spy_min,
            "strategy_rows":     rows,
            "spy_rows":          spy_rows,
            "start_year":        start_year,
            "initial_capital":   init,
        })
        st.rerun()

    # -- Chart --
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

    spy_by_year = spy_by_year_ret

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

    spy_values_by_year = spy_val_by_year

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

# ---------------------------------------------------------------------------
# Comparison view — driven by sidebar checkboxes + Compare button
# ---------------------------------------------------------------------------

saved = st.session_state.get("saved_backtests", [])
_cmp_indices = st.session_state.get("_compare_indices", [])
chosen = [saved[i] for i in _cmp_indices if i < len(saved)]
chosen_labels = [s["name"][:45] for s in chosen]

if len(chosen) >= 2:
    st.divider()
    st.subheader("Side-by-side comparison")

    # Metric comparison table
    _metrics = [
        ("Final Portfolio Value", lambda s: f"${s['final']:,.0f}"),
        ("Total ROI",             lambda s: f"{s['total_roi']*100:+.1f}%"),
        ("S&P 500 Final Value",   lambda s: f"${s['spy_final']:,.0f}"),
        ("S&P 500 ROI",           lambda s: f"{s['spy_roi']*100:+.1f}%"),
        ("Years Beat S&P",        lambda s: f"{s['years_beat']}/{s['total_years']} ({s['years_beat']/s['total_years']*100:.0f}%)"),
        ("Longest Losing Streak", lambda s: f"{s['max_behind']} yr{'s' if s['max_behind']!=1 else ''}"),
        ("Portfolio Low",         lambda s: f"${s['port_min']:,.0f}"),
        ("S&P Low",               lambda s: f"${s['spy_min']:,.0f}"),
    ]

    _hdr = "".join(f"<th style='text-align:right;padding:8px 12px;'>{lbl[:40]}</th>" for lbl in chosen_labels)
    _cmp_rows = ""
    for _label, _fn in _metrics:
        _vals = [_fn(s) for s in chosen]
        _cmp_rows += f"<tr><td style='padding:6px 12px;font-weight:600;'>{_label}</td>"
        for v in _vals:
            _cmp_rows += f"<td style='padding:6px 12px;text-align:right;'>{v}</td>"
        _cmp_rows += "</tr>"

    st.markdown(f"""
    <style>.saved-table{{border-collapse:collapse;width:100%;font-size:0.91em;}}
    .saved-table th{{font-weight:600;padding:8px 12px;border-bottom:2px solid #ddd;text-align:left;}}
    .saved-table td{{border-bottom:1px solid #eee;}}
    .saved-table tr:hover td{{background:rgba(0,0,0,0.03);}}</style>
    <table class='saved-table'><thead><tr>
      <th>Metric</th>{_hdr}
    </tr></thead><tbody>{_cmp_rows}</tbody></table>
    """, unsafe_allow_html=True)

    # Overlay chart
    st.markdown("#### Portfolio value over time")
    _colors = ["#00e676", "#ff9800", "#e91e63"]
    fig_cmp = go.Figure()
    for i, s in enumerate(chosen):
        _yrs = [r["year"] for r in s["strategy_rows"]]
        _vals = [r["end_value"] for r in s["strategy_rows"]]
        fig_cmp.add_trace(go.Scatter(
            x=_yrs, y=_vals,
            mode="lines+markers",
            name=s["name"][:40],
            line=dict(color=_colors[i], width=2.5),
            marker=dict(size=5),
            hovertemplate="Year: %{x}<br>$%{y:,.0f}<extra></extra>",
        ))
    fig_cmp.add_trace(go.Scatter(
        x=[r["year"] for r in chosen[0]["spy_rows"]],
        y=[r["value"] for r in chosen[0]["spy_rows"]],
        mode="lines", name="S&P 500",
        line=dict(color="#2196f3", width=2, dash="dot"),
        hovertemplate="Year: %{x}<br>SPY: $%{y:,.0f}<extra></extra>",
    ))
    fig_cmp.update_layout(
        template="plotly_dark", hovermode="x unified", height=420,
        yaxis=dict(tickformat="$,.0f"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig_cmp, width="stretch")
