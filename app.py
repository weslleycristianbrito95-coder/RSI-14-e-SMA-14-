from datetime import datetime, timedelta
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pytz
import streamlit as st
import yfinance as yf

st.set_page_config(
    page_title="Viés Macro 15M & 5M — RSI vs SMA",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    [data-testid="stMetricValue"] { font-size: 1.35rem !important; }
    [data-testid="stMetricLabel"] { font-size: 0.82rem !important; }
    [data-testid="stMetricDelta"] { font-size: 0.80rem !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------------
# Universo
# ----------------------------------------------------------------------------
TICKER_GROUPS: Dict[str, list] = {
    "Índices & ETFs": [
        "^GSPC", "^DJI", "^RUT", "^NDX",
        "SPY", "QQQ", "SMH", "IGV", "XLK", "XLC", "XLY", "XLF",
    ],
    "Ações": ["AAPL", "NVDA", "MSFT", "AMZN", "MU", "AMD", "TSLA"],
    "Forex": [
        "EURUSD=X", "GBPUSD=X", "USDJPY=X", "AUDUSD=X", "NZDUSD=X",
        "AUDJPY=X", "NZDJPY=X", "USDCHF=X", "MXNUSD=X",
    ],
    "Cripto": ["BTC-USD", "ETH-USD", "SOL-USD", "ICP-USD", "RENDER-USD", "FET-USD", "NEAR-USD"],
    "Volatilidade & Macro": ["^VIX", "DX-Y.NYB"],
}
TICKERS = [t for group in TICKER_GROUPS.values() for t in group]
GROUP_OF = {t: g for g, ts in TICKER_GROUPS.items() for t in ts}

# Ativos lidos de forma invertida (queda do ativo = viés de alta para risco)
INVERTED_TICKERS = {"^VIX", "DX-Y.NYB"}
LABELS = {
    "^GSPC": "US500",
    "^DJI": "US30",
    "^RUT": "US2000",
    "^NDX": "USTECH100",
    "^VIX": "VIX (inv)",
    "DX-Y.NYB": "DXY (inv)",
}

RSI_PERIOD = 14
RSI_MA_PERIOD = 20
Z_WINDOW = 20
TIMEZONE_BRT = pytz.timezone("America/Sao_Paulo")

UP, DOWN, AMBER = "#10b981", "#ef4444", "#f59e0b"


def label(ticker: str) -> str:
    return LABELS.get(ticker, ticker.replace("=X", ""))


def stretch(fn, *args, **kwargs):
    """Compatível com versões novas (width='stretch') e antigas (use_container_width)."""
    try:
        return fn(*args, width="stretch", **kwargs)
    except Exception:
        return fn(*args, use_container_width=True, **kwargs)


# ----------------------------------------------------------------------------
# Cálculos
# ----------------------------------------------------------------------------
def calculate_rsi_wilder(series: pd.Series, period: int = 14) -> pd.Series:
    """RSI de Wilder. Mantém NaN no aquecimento; RSI=100 se não há perdas; 50 se não há movimento."""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()

    with np.errstate(divide="ignore", invalid="ignore"):
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))  # avg_loss == 0 -> rs = inf -> RSI = 100

    flat = (avg_gain == 0) & (avg_loss == 0)
    return rsi.where(~flat, 50.0)


def zscore_last(series: pd.Series, window: int = Z_WINDOW) -> float:
    """Z-score da última observação contra as `window` observações ANTERIORES (sem incluí-la)."""
    s = series.replace([np.inf, -np.inf], np.nan).dropna()
    if len(s) < window + 1:
        return np.nan
    base = s.iloc[-(window + 1):-1]
    sd = base.std(ddof=1)
    if not np.isfinite(sd) or sd <= 0:
        return np.nan
    return float((s.iloc[-1] - base.mean()) / sd)


def _extract_ticker_df(raw_data: pd.DataFrame, ticker: str, total_tickers: int) -> pd.DataFrame:
    if raw_data is None or raw_data.empty:
        return pd.DataFrame()
    if total_tickers == 1:
        return raw_data.copy()
    if isinstance(raw_data.columns, pd.MultiIndex):
        if ticker in raw_data.columns.get_level_values(0):
            return raw_data[ticker].copy()
        if ticker in raw_data.columns.get_level_values(-1):
            return raw_data.xs(ticker, axis=1, level=-1).copy()
    return pd.DataFrame()


def _process_intraday_rsi(
    raw_df: pd.DataFrame,
    ticker: str,
    total_tickers: int,
    rsi_len: int,
    ma_len: int,
    closed_only: bool,
    tf_minutes: int,
    now_brt: pd.Timestamp,
) -> pd.DataFrame:
    """Processa série intraday (15m ou 5m) e retorna DataFrame com RSI e SMA."""
    df = _extract_ticker_df(raw_df, ticker, total_tickers)
    if df.empty or "Close" not in df.columns:
        return pd.DataFrame()
    df = df.dropna(subset=["Close"]).copy()
    if df.empty:
        return pd.DataFrame()

    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC").tz_convert(TIMEZONE_BRT)
    else:
        df.index = df.index.tz_convert(TIMEZONE_BRT)

    if closed_only:
        df = df[df.index <= now_brt - timedelta(minutes=tf_minutes)]

    sign = -1.0 if ticker in INVERTED_TICKERS else 1.0
    rsi = calculate_rsi_wilder(df["Close"], period=rsi_len)
    if sign < 0:
        rsi = 100.0 - rsi

    df["RSI"] = rsi
    df["RSI_MA"] = rsi.rolling(window=ma_len).mean()
    df = df.dropna(subset=["RSI", "RSI_MA"])
    if len(df) < 2:
        return pd.DataFrame()

    df["Spread"] = df["RSI"] - df["RSI_MA"]
    df["Acima_Media"] = df["RSI"] > df["RSI_MA"]
    return df


@st.cache_data(ttl=25, show_spinner=False)
def fetch_and_process_data(
    tickers: Tuple[str, ...],
    rsi_len: int,
    ma_len: int,
    include_extended: bool,
    closed_only: bool,
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, bool]]:
    try:
        raw_daily = yf.download(
            tickers=list(tickers), period="60d", interval="1d",
            group_by="ticker", threads=True, progress=False,
        )
        raw_15m = yf.download(
            tickers=list(tickers), period="10d", interval="15m",
            group_by="ticker", prepost=include_extended, threads=True, progress=False,
        )
        raw_5m = yf.download(
            tickers=list(tickers), period="5d", interval="5m",
            group_by="ticker", prepost=include_extended, threads=True, progress=False,
        )
    except Exception:
        return {}, {}

    processed_15m: Dict[str, pd.DataFrame] = {}
    status_5m: Dict[str, bool] = {}
    now_brt = pd.Timestamp.now(tz=TIMEZONE_BRT)
    n_tickers = len(tickers)

    for ticker in tickers:
        try:
            sign = -1.0 if ticker in INVERTED_TICKERS else 1.0

            # --- Dados Diários: Variação % e Z-Scores ---
            z_ret, z_amp, daily_chg = np.nan, np.nan, np.nan
            df_d = _extract_ticker_df(raw_daily, ticker, n_tickers)
            if not df_d.empty and "Close" in df_d.columns:
                df_d = df_d.dropna(subset=["Close"])
                if len(df_d) >= 2:
                    prev_close = float(df_d["Close"].iloc[-2])
                    last_close = float(df_d["Close"].iloc[-1])
                    if prev_close != 0:
                        daily_chg = ((last_close / prev_close) - 1.0) * 100.0

                z_ret = zscore_last(df_d["Close"].pct_change() * 100.0 * sign)
                if {"High", "Low"}.issubset(df_d.columns):
                    low = df_d["Low"].replace(0, np.nan)
                    z_amp = zscore_last((df_d["High"] - df_d["Low"]) / low)

            # --- Status 5m ---
            df_5 = _process_intraday_rsi(
                raw_5m, ticker, n_tickers, rsi_len, ma_len, closed_only, 5, now_brt
            )
            if not df_5.empty:
                status_5m[ticker] = bool(df_5.iloc[-1]["Acima_Media"])

            # --- Intraday 15m ---
            df_15 = _process_intraday_rsi(
                raw_15m, ticker, n_tickers, rsi_len, ma_len, closed_only, 15, now_brt
            )
            if df_15.empty:
                continue

            df_15["Var_Dia"] = daily_chg
            df_15["Z_Score_20d"] = z_ret
            df_15["Z_Score_Amp_20d"] = z_amp
            processed_15m[ticker] = df_15
        except Exception:
            continue

    return processed_15m, status_5m


# ----------------------------------------------------------------------------
# Gráficos
# ----------------------------------------------------------------------------
def fmt_price(ticker: str, price: float) -> str:
    if ticker.endswith("=X") or ticker == "DX-Y.NYB":
        return f"{price:.{3 if 'JPY' in ticker else 4}f}"
    if ticker.startswith("^"):
        return f"{price:,.2f}"
    return f"${price:.4f}" if price < 1 else f"${price:,.2f}"


def chart_title(ticker: str, df: pd.DataFrame) -> str:
    last = df.iloc[-1]
    above = bool(last["Acima_Media"])
    color = UP if above else DOWN
    status = "▲ ACIMA" if above else "▼ ABAIXO"
    z_amp = last["Z_Score_Amp_20d"]
    z_txt = f"{z_amp:+.1f}" if pd.notna(z_amp) else "–"
    return (
        f"<b>{label(ticker)}</b>  <span style=\"color:{color}\">{status}</span>"
        f" · RSI {last['RSI']:.0f} · ZAmp {z_txt}"
    )


def tick_config(x_labels: np.ndarray, n_ticks: int = 5):
    idx = np.unique(np.linspace(0, len(x_labels) - 1, n_ticks).round().astype(int))
    vals, texts, prev_day = [], [], None
    for i in idx:
        day, hm = x_labels[i].split(" ")
        texts.append(f"{day}<br>{hm}" if day != prev_day else hm)
        vals.append(x_labels[i])
        prev_day = day
    return vals, texts


def build_group_figure(tickers, data_map, candles: int, ncols: int) -> go.Figure:
    n = len(tickers)
    nrows = int(np.ceil(n / ncols))
    fig = make_subplots(
        rows=nrows, cols=ncols,
        subplot_titles=[chart_title(t, data_map[t]) for t in tickers],
        vertical_spacing=0.2 / nrows, horizontal_spacing=0.05,
    )

    for idx, ticker in enumerate(tickers):
        row, col = idx // ncols + 1, idx % ncols + 1
        d = data_map[ticker].tail(candles)
        x = np.array(d.index.strftime("%d/%m %H:%M"))
        rsi, ma = d["RSI"].to_numpy(), d["RSI_MA"].to_numpy()
        above = rsi > ma
        color = UP if above[-1] else DOWN

        # zonas e referências
        fig.add_hrect(y0=70, y1=100, fillcolor="rgba(239,68,68,0.07)", line_width=0, layer="below", row=row, col=col)
        fig.add_hrect(y0=0, y1=30, fillcolor="rgba(16,185,129,0.07)", line_width=0, layer="below", row=row, col=col)
        for lvl, dash in ((70, "dash"), (30, "dash"), (50, "dot")):
            fig.add_hline(y=lvl, line_dash=dash, line_width=1, line_color="rgba(255,255,255,0.18)", row=row, col=col)

        # SMA + preenchimento verde (RSI > SMA) e vermelho (RSI < SMA) entre as duas curvas
        fig.add_trace(go.Scatter(
            x=x, y=ma, mode="lines", name="SMA",
            line=dict(color=AMBER, width=1.4, dash="dot"),
            hovertemplate="SMA %{y:.1f}<extra></extra>"), row=row, col=col)
        fig.add_trace(go.Scatter(
            x=x, y=np.where(above, rsi, ma), mode="lines", line=dict(width=0),
            fill="tonexty", fillcolor="rgba(16,185,129,0.28)", hoverinfo="skip"), row=row, col=col)
        fig.add_trace(go.Scatter(
            x=x, y=ma, mode="lines", line=dict(width=0), hoverinfo="skip"), row=row, col=col)
        fig.add_trace(go.Scatter(
            x=x, y=np.where(above, ma, rsi), mode="lines", line=dict(width=0),
            fill="tonexty", fillcolor="rgba(239,68,68,0.28)", hoverinfo="skip"), row=row, col=col)

        # RSI colorido por trecho: verde quando RSI > SMA, vermelho quando RSI < SMA
        for state, seg_color in ((True, UP), (False, DOWN)):
            seg = above[1:] == state
            k = int(seg.sum())
            if k == 0:
                continue
            xs = np.empty(3 * k, dtype=object)
            ys = np.full(3 * k, np.nan)
            xs[0::3], xs[1::3], xs[2::3] = x[:-1][seg], x[1:][seg], x[1:][seg]
            ys[0::3], ys[1::3] = rsi[:-1][seg], rsi[1:][seg]
            fig.add_trace(go.Scatter(
                x=xs, y=ys, mode="lines", connectgaps=False,
                line=dict(color=seg_color, width=2), hoverinfo="skip"), row=row, col=col)

        # camada invisível só para o hover mostrar o valor do RSI
        fig.add_trace(go.Scatter(
            x=x, y=rsi, mode="markers", name="RSI",
            marker=dict(size=1, opacity=0),
            hovertemplate="RSI %{y:.1f}<extra></extra>"), row=row, col=col)

        # último ponto
        fig.add_trace(go.Scatter(
            x=[x[-1]], y=[rsi[-1]], mode="markers", hoverinfo="skip",
            marker=dict(size=8, color=color, line=dict(color="#fff", width=1.2))), row=row, col=col)

        vals, texts = tick_config(x)
        fig.update_xaxes(
            type="category", categoryorder="array", categoryarray=list(x),
            tickmode="array", tickvals=vals, ticktext=texts,
            tickfont=dict(size=10), showgrid=False, row=row, col=col)
        fig.update_yaxes(
            range=[0, 100], tickvals=[30, 50, 70], tickfont=dict(size=10),
            gridcolor="rgba(255,255,255,0.05)", zeroline=False, row=row, col=col)

    fig.update_layout(
        height=275 * nrows,
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.02)",
        margin=dict(l=10, r=10, t=45, b=10),
        hovermode="x unified",
        showlegend=False,
        font=dict(size=11),
    )
    fig.update_annotations(font_size=12)
    return fig


# ----------------------------------------------------------------------------
# Dashboard
# ----------------------------------------------------------------------------
def render_dashboard(rsi_len, ma_len, include_extended, closed_only, candles_chart, ncols) -> None:
    with st.spinner("Atualizando cotações..."):
        data_map, status_5m = fetch_and_process_data(
            tuple(TICKERS), int(rsi_len), int(ma_len), include_extended, closed_only
        )

    if not data_map:
        st.error("Não foi possível obter dados do Yahoo Finance no momento. Tente novamente em instantes.")
        return

    rows, acima_15m, z_pos, z_neg = [], 0, 0, 0
    rsi_col, ma_col = f"RSI ({rsi_len})", f"Média RSI ({ma_len})"

    for ticker in TICKERS:
        df = data_map.get(ticker)
        if df is None:
            continue
        last, prev = df.iloc[-1], df.iloc[-2]
        is_above, was_above = bool(last["Acima_Media"]), bool(prev["Acima_Media"])
        acima_15m += is_above

        z_val = float(last["Z_Score_20d"]) if pd.notna(last["Z_Score_20d"]) else np.nan
        if pd.notna(z_val):
            z_pos += z_val > 0
            z_neg += z_val < 0
        z_amp = float(last["Z_Score_Amp_20d"]) if pd.notna(last["Z_Score_Amp_20d"]) else np.nan
        var_dia = float(last["Var_Dia"]) if pd.notna(last["Var_Dia"]) else np.nan

        if is_above and not was_above:
            evento = "CRUZOU PARA CIMA"
        elif not is_above and was_above:
            evento = "CRUZOU PARA BAIXO"
        else:
            evento = "Mantendo tendência"

        st_5m_bool = status_5m.get(ticker)
        if st_5m_bool is True:
            st_5m_txt = "ACIMA"
        elif st_5m_bool is False:
            st_5m_txt = "ABAIXO"
        else:
            st_5m_txt = "-"

        rows.append({
            "Grupo": GROUP_OF[ticker],
            "Ativo": label(ticker),
            "Último Candle (BRT)": df.index[-1].strftime("%d/%m %H:%M"),
            "Preço": fmt_price(ticker, float(last["Close"])),
            "Var. Dia (%)": var_dia,
            rsi_col: float(last["RSI"]),
            ma_col: float(last["RSI_MA"]),
            "Spread (RSI - MA)": float(last["Spread"]),
            "Z-Score (20d)": z_val,
            "Z-Score Amp (20d)": z_amp,
            "Status 5M": st_5m_txt,
            "Status 15M": "ACIMA DA MÉDIA" if is_above else "ABAIXO DA MÉDIA",
            "Sinal Último Candle": evento,
        })

    summary_df = pd.DataFrame(rows)
    total_15m = len(summary_df)
    abaixo_15m = total_15m - acima_15m

    # Contagem 5M
    total_5m = len(status_5m)
    acima_5m = sum(1 for v in status_5m.values() if v)
    abaixo_5m = total_5m - acima_5m

    if z_pos > z_neg:
        z_winner = f"POS ({z_pos} vs {z_neg}) ✅"
    elif z_neg > z_pos:
        z_winner = f"NEG ({z_neg} vs {z_pos}) 🔻"
    else:
        z_winner = f"EMP ({z_pos} vs {z_neg}) ⚖️"

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("15M: Acima da Média", f"{acima_15m} de {total_15m}")
    c2.metric("15M: Abaixo da Média", f"{abaixo_15m} de {total_15m}")
    c3.metric(
        "5M: Acima vs Abaixo SMA",
        f"▲ {acima_5m} | ▼ {abaixo_5m}",
        delta=f"Total: {total_5m} ativos",
        delta_color="off",
    )
    c4.metric("Dominância Z-Score (20d)", z_winner)
    c5.metric("Última Atualização (BRT)", datetime.now(TIMEZONE_BRT).strftime("%H:%M:%S"))

    missing = [label(t) for t in TICKERS if t not in data_map]
    if missing:
        st.caption(f"Sem dados suficientes agora: {', '.join(missing)}")

    def hl_status(v):
        if v in ("ACIMA DA MÉDIA", "ACIMA"):
            return f"background-color: rgba(16,185,129,0.25); color: {UP}; font-weight: bold;"
        if v in ("ABAIXO DA MÉDIA", "ABAIXO"):
            return f"background-color: rgba(239,68,68,0.25); color: {DOWN}; font-weight: bold;"
        return ""

    def hl_var(v):
        if pd.isna(v) or v == 0:
            return ""
        if v > 0:
            return f"color: {UP}; font-weight: bold;"
        return f"color: {DOWN}; font-weight: bold;"

    def hl_event(v):
        if "CIMA" in v:
            return f"color: {UP}; font-weight: bold;"
        if "BAIXO" in v:
            return f"color: {DOWN}; font-weight: bold;"
        return "color: #888888;"

    def hl_z(v):
        if pd.isna(v):
            return ""
        if v > 1.5:
            return f"background-color: rgba(16,185,129,0.2); color: {UP}; font-weight: bold;"
        if v < -1.5:
            return f"background-color: rgba(239,68,68,0.2); color: {DOWN}; font-weight: bold;"
        return ""

    styled = (
        summary_df.style
        .map(hl_status, subset=["Status 15M", "Status 5M"])
        .map(hl_var, subset=["Var. Dia (%)"])
        .map(hl_event, subset=["Sinal Último Candle"])
        .map(hl_z, subset=["Z-Score (20d)", "Z-Score Amp (20d)"])
        .format({
            "Var. Dia (%)": "{:+.2f}%",
            rsi_col: "{:.2f}",
            ma_col: "{:.2f}",
            "Spread (RSI - MA)": "{:+.2f}",
            "Z-Score (20d)": "{:+.2f}",
            "Z-Score Amp (20d)": "{:+.2f}",
        }, na_rep="-")
    )
    stretch(st.dataframe, styled, hide_index=True)
    st.caption(
        "VIX e DXY: RSI e Z-Score de retorno calculados com sinal invertido (queda = viés de alta). "
        "Var. Dia (%) reflete a variação nominal do ativo."
    )

    st.subheader(f"RSI({rsi_len}) vs. Média ({ma_len}) — Gráfico de 15 Minutos")
    tabs = st.tabs(list(TICKER_GROUPS))
    for tab, (group, tks) in zip(tabs, TICKER_GROUPS.items()):
        with tab:
            active = [t for t in tks if t in data_map]
            if not active:
                st.info("Sem dados para este grupo no momento.")
                continue
            fig = build_group_figure(active, data_map, candles_chart, ncols)
            stretch(st.plotly_chart, fig, key=f"chart_{group}", config={"displayModeBar": False})


def main() -> None:
    st.title("VIÉS MACRO 15M & 5M")

    with st.sidebar:
        st.header("Parâmetros")
        rsi_len = st.number_input("Período RSI", min_value=2, max_value=50, value=RSI_PERIOD)
        ma_len = st.number_input("Média Móvel no RSI (SMA)", min_value=2, max_value=50, value=RSI_MA_PERIOD)
        include_extended = st.checkbox("Incluir Pré / After-Market", value=False)
        closed_only = st.checkbox("Somente candles fechados (sem repaint)", value=True)
        auto_refresh = st.checkbox("Auto-atualizar (30s)", value=True)
        candles_chart = st.slider("Candles visíveis no gráfico", min_value=30, max_value=150, value=78)
        ncols = st.slider("Colunas de gráficos", min_value=2, max_value=4, value=3)

        if stretch(st.button, "Atualizar Agora"):
            st.cache_data.clear()

    args = (int(rsi_len), int(ma_len), include_extended, closed_only, candles_chart, ncols)

    if auto_refresh:
        @st.fragment(run_every=30)
        def auto_updating_view():
            render_dashboard(*args)

        auto_updating_view()
    else:
        render_dashboard(*args)


if __name__ == "__main__":
    main()
