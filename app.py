from datetime import datetime
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pytz
import streamlit as st
import yfinance as yf

st.set_page_config(
    page_title="Scanner Intraday RSI vs SMA",
    layout="wide",
    initial_sidebar_state="expanded",
)

# CSS customizado para reduzir o tamanho da fonte das métricas (KPIs) no topo
st.markdown(
    """
    <style>
    [data-testid="stMetricValue"] {
        font-size: 1.4rem !important;
    }
    [data-testid="stMetricLabel"] {
        font-size: 0.85rem !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Lista de ativos organizada por segmento de mercado
TICKERS = [
    # --- ÍNDICES & ETFS DE SETORES ---
    "QQQ",        # Nasdaq 100
    "SMH",        # VanEck Semiconductor ETF
    "IGV",        # iShares Expanded Tech-Software ETF
    "XLK",        # Technology Select Sector SPDR
    "XLC",        # Communication Services Select Sector
    "XLY",        # Consumer Discretionary Select Sector
    "XLF",        # Financial Select Sector SPDR

    # --- AÇÕES LÍDERES (TECNOLOGIA / CHIPS) ---
    "AAPL",       # Apple
    "NVDA",       # NVIDIA
    "MSFT",       # Microsoft
    "AMZN",       # Amazon
    "MU",         # Micron Technology
    "AMD",        # Advanced Micro Devices
    "TSLA",       # Tesla

    # --- PARES DE MOEDAS (FOREX) ---
    "EURUSD=X",   # Euro / Dólar Americano
    "GBPUSD=X",   # Libra Esterlina / Dólar Americano
    "USDJPY=X",   # Dólar Americano / Iene Japonês
    "AUDUSD=X",   # Dólar Australiano / Dólar Americano
    "NZDUSD=X",   # Dólar Neozelandês / Dólar Americano
    "AUDJPY=X",   # Dólar Australiano / Iene Japonês
    "NZDJPY=X",   # Dólar Neozelandês / Iene Japonês
    "USDCHF=X",   # Dólar Americano / Franco Suíço
    "MXNUSD=X",   # Peso Mexicano / Dólar Americano

    # --- CRIPTOMOEDAS ---
    "BTC-USD",    # Bitcoin
    "ETH-USD",    # Ethereum
    "SOL-USD",    # Solana
    "ICP-USD",    # Internet Computer
    "RENDER-USD", # Render
    "FET-USD",    # Artificial Superintelligence Alliance (Fetch.ai)
    "NEAR-USD",   # NEAR Protocol

    # --- VOLATILIDADE ---
    "^VIX",       # Índice de Volatilidade CBOE

    # --- MACRO & RENDA FIXA ---
    "DX-Y.NYB",   # Índice Dólar (DXY)
]

RSI_PERIOD = 14
RSI_MA_PERIOD = 20
TIMEZONE_BRT = pytz.timezone("America/Sao_Paulo")

# Ativos que operam com a lógica e escala invertidas
INVERTED_TICKERS = ["^VIX", "DX-Y.NYB"]


def calculate_rsi_wilder(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50.0)


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


@st.cache_data(ttl=25, show_spinner=False)
def fetch_and_process_data(
    tickers: Tuple[str, ...],
    rsi_len: int,
    ma_len: int,
    include_extended: bool,
) -> Dict[str, pd.DataFrame]:
    raw_daily = yf.download(
        tickers=list(tickers),
        period="40d",
        interval="1d",
        group_by="ticker",
        threads=True,
        progress=False,
    )

    raw_intraday = yf.download(
        tickers=list(tickers),
        period="5d",
        interval="15m",
        group_by="ticker",
        prepost=include_extended,
        threads=True,
        progress=False,
    )

    processed: Dict[str, pd.DataFrame] = {}

    for ticker in tickers:
        try:
            # Processa dados diários para Z-Score de retorno (20d) e amplitude (20d)
            df_d = _extract_ticker_df(raw_daily, ticker, len(tickers))
            df_d = df_d.dropna(subset=["Close"])

            z_score_val = np.nan
            if len(df_d) >= 21:
                df_d["Daily_Return"] = df_d["Close"].pct_change()
                rolling_mean = df_d["Daily_Return"].rolling(20).mean()
                rolling_std = df_d["Daily_Return"].rolling(20).std()

                latest_return = float(df_d["Daily_Return"].iloc[-1])
                l_mean = float(rolling_mean.iloc[-1])
                l_std = float(rolling_std.iloc[-1])

                if pd.notna(l_std) and l_std > 0 and pd.notna(latest_return) and pd.notna(l_mean):
                    z_score_val = (latest_return - l_mean) / l_std

            z_score_amp_val = np.nan
            if {"High", "Low"}.issubset(df_d.columns):
                df_d_amp = df_d.dropna(subset=["High", "Low"]).copy()
                low_safe = df_d_amp["Low"].replace(0, np.nan)
                df_d_amp["Daily_Amp"] = (df_d_amp["High"] - df_d_amp["Low"]) / low_safe

                if len(df_d_amp) >= 20:
                    amp_mean = df_d_amp["Daily_Amp"].rolling(20).mean()
                    amp_std = df_d_amp["Daily_Amp"].rolling(20).std()

                    latest_amp = float(df_d_amp["Daily_Amp"].iloc[-1])
                    a_mean = float(amp_mean.iloc[-1])
                    a_std = float(amp_std.iloc[-1])

                    if pd.notna(a_std) and a_std > 0 and pd.notna(latest_amp) and pd.notna(a_mean):
                        z_score_amp_val = (latest_amp - a_mean) / a_std

            # Processa dados de 15 minutos
            df = _extract_ticker_df(raw_intraday, ticker, len(tickers))
            df = df.dropna(subset=["Close"])

            if df.empty or len(df) < (rsi_len + ma_len + 2):
                continue

            if df.index.tz is None:
                df.index = df.index.tz_localize("UTC").tz_convert(TIMEZONE_BRT)
            else:
                df.index = df.index.tz_convert(TIMEZONE_BRT)

            rsi_series = calculate_rsi_wilder(df["Close"], period=rsi_len)

            df["RSI"] = rsi_series
            df["RSI_MA"] = df["RSI"].rolling(window=ma_len).mean()
            df["Spread"] = df["RSI"] - df["RSI_MA"]
            df["Acima_Media"] = df["RSI"] > df["RSI_MA"]
            df["Z_Score_20d"] = z_score_val
            df["Z_Score_Amp_20d"] = z_score_amp_val

            processed[ticker] = df
        except Exception as exc:
            st.warning(f"Erro ao processar {ticker}: {exc}")

    return processed


def render_dashboard(
    rsi_len: int,
    ma_len: int,
    include_extended: bool,
    candles_chart: int,
) -> None:
    with st.spinner("Atualizando cotações..."):
        data_map = fetch_and_process_data(
            tuple(TICKERS), int(rsi_len), int(ma_len), include_extended
        )

    if not data_map:
        st.error("Não foi possível obter dados do Yahoo Finance no momento. Tente novamente em instantes.")
        return

    summary_rows = []
    acima_count = 0
    z_positive_count = 0
    z_negative_count = 0

    rsi_col_name = f"RSI ({rsi_len})"
    ma_col_name = f"Média RSI ({ma_len})"

    for ticker in TICKERS:
        df = data_map.get(ticker)
        if df is None:
            continue

        last = df.iloc[-1]
        prev = df.iloc[-2]

        is_above = bool(last["Acima_Media"])
        was_above = bool(prev["Acima_Media"])

        if ticker in INVERTED_TICKERS:
            display_above = not is_above
            display_was_above = not was_above
        else:
            display_above = is_above
            display_was_above = was_above

        if display_above:
            acima_count += 1

        z_val = float(last["Z_Score_20d"]) if pd.notna(last["Z_Score_20d"]) else np.nan
        if pd.notna(z_val):
            if z_val > 0:
                z_positive_count += 1
            elif z_val < 0:
                z_negative_count += 1

        z_amp_val = float(last["Z_Score_Amp_20d"]) if pd.notna(last["Z_Score_Amp_20d"]) else np.nan

        if display_above and not display_was_above:
            evento = "CRUZOU PARA CIMA"
        elif not display_above and display_was_above:
            evento = "CRUZOU PARA BAIXO"
        else:
            evento = "Mantendo tendência"

        summary_rows.append(
            {
                "Ativo": ticker,
                "Ticker_Original": ticker,
                "Último Candle (BRT)": df.index[-1].strftime("%d/%m %H:%M"),
                "Preço": round(float(last["Close"]), 4) if "=X" in ticker or "DX" in ticker else round(float(last["Close"]), 2),
                rsi_col_name: round(float(last["RSI"]), 2),
                ma_col_name: round(float(last["RSI_MA"]), 2),
                "Spread (RSI - MA)": round(float(last["Spread"]), 2),
                "Z-Score (20d)": z_val,
                "Z-Score Amp (20d)": z_amp_val,
                "Status": "ACIMA DA MÉDIA" if display_above else "ABAIXO DA MÉDIA",
                "Sinal Último Candle": evento,
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    total_ativos = len(summary_df)

    if z_positive_count > z_negative_count:
        z_winner = f"POS ({z_positive_count} vs {z_negative_count}) ✅"
    elif z_negative_count > z_positive_count:
        z_winner = f"NEG ({z_negative_count} vs {z_positive_count}) 🔻"
    else:
        z_winner = f"EMP ({z_positive_count} vs {z_negative_count}) ⚖️"

    now_brt = datetime.now(TIMEZONE_BRT).strftime("%H:%M:%S")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Ativos Acima da Média (RSI > SMA)", f"{acima_count} de {total_ativos}")
    col2.metric("Ativos Abaixo da Média (RSI < SMA)", f"{total_ativos - acima_count} de {total_ativos}")
    col3.metric("Dominância Z-Score (20d)", z_winner)
    col4.metric("Última Atualização (Brasília)", now_brt)

    def highlight_status(val: str) -> str:
        if val == "ACIMA DA MÉDIA":
            return "background-color: rgba(16, 185, 129, 0.25); color: #10b981; font-weight: bold;"
        if val == "ABAIXO DA MÉDIA":
            return "background-color: rgba(239, 68, 68, 0.25); color: #ef4444; font-weight: bold;"
        return ""

    def highlight_event(val: str) -> str:
        if "CIMA" in val:
            return "color: #10b981; font-weight: bold;"
        if "BAIXO" in val:
            return "color: #ef4444; font-weight: bold;"
        return "color: #888888;"

    def highlight_zscore(val: float) -> str:
        if pd.isna(val):
            return ""
        if val > 1.5:
            return "background-color: rgba(16, 185, 129, 0.2); color: #10b981; font-weight: bold;"
        elif val < -1.5:
            return "background-color: rgba(239, 68, 68, 0.2); color: #ef4444; font-weight: bold;"
        return ""

    formatted_prices = []
    for _, row in summary_df.iterrows():
        tick = str(row["Ticker_Original"])
        preco_val = row["Preço"]
        if "=X" in tick or "DX" in tick:
            formatted_prices.append(f"{preco_val:.4f}")
        else:
            formatted_prices.append(f"${preco_val:.2f}")
    summary_df["Preço Formated"] = formatted_prices

    cols_order = [
        "Ativo", "Último Candle (BRT)", "Preço Formated", rsi_col_name,
        ma_col_name, "Spread (RSI - MA)", "Z-Score (20d)", "Z-Score Amp (20d)",
        "Status", "Sinal Último Candle"
    ]
    display_df = summary_df[cols_order].rename(columns={"Preço Formated": "Preço"})

    styled_df = (
        display_df.style
        .map(highlight_status, subset=["Status"])
        .map(highlight_event, subset=["Sinal Último Candle"])
        .map(highlight_zscore, subset=["Z-Score (20d)"])
        .map(highlight_zscore, subset=["Z-Score Amp (20d)"])
        .format(
            {
                rsi_col_name: "{:.2f}",
                ma_col_name: "{:.2f}",
                "Spread (RSI - MA)": "{:+.2f}",
                "Z-Score (20d)": "{:+.2f}",
                "Z-Score Amp (20d)": "{:+.2f}",
            },
            na_rep="-",
        )
    )

    st.dataframe(styled_df, use_container_width=True, hide_index=True)

    st.subheader(f"Curva RSI({rsi_len}) vs. Média Móvel ({ma_len}) — Gráfico de 15 Minutos")

    active_tickers = [t for t in TICKERS if t in data_map]
    num_plots = len(active_tickers)
    num_cols = 2
    num_rows = (num_plots + 1) // num_cols

    subplot_titles_list = []
    for t in active_tickers:
        is_abv = bool(data_map[t]["Acima_Media"].iloc[-1])
        z_amp_last = data_map[t]["Z_Score_Amp_20d"].iloc[-1]
        z_amp_txt = f"Z-Amp: {z_amp_last:+.2f}" if pd.notna(z_amp_last) else "Z-Amp: -"
        if t in INVERTED_TICKERS:
            label = "ABAIXO" if is_abv else "ACIMA"
            subplot_titles_list.append(f"{t} — {label} (Invertido) | {z_amp_txt}")
        else:
            label = "ACIMA" if is_abv else "ABAIXO"
            subplot_titles_list.append(f"{t} — {label} | {z_amp_txt}")

    fig = make_subplots(
        rows=num_rows,
        cols=num_cols,
        subplot_titles=subplot_titles_list,
        vertical_spacing=0.04,
        horizontal_spacing=0.06,
    )

    for idx, ticker in enumerate(active_tickers):
        row = (idx // num_cols) + 1
        col = (idx % num_cols) + 1
        df_plot = data_map[ticker].tail(candles_chart)
        x_labels = df_plot.index.strftime("%d/%m %H:%M")

        is_above = bool(df_plot["Acima_Media"].iloc[-1])

        if ticker in INVERTED_TICKERS:
            rsi_color = "#ef4444" if is_above else "#10b981"
        else:
            rsi_color = "#10b981" if is_above else "#ef4444"

        fig.add_trace(
            go.Scatter(
                x=x_labels,
                y=df_plot["RSI"],
                mode="lines",
                name=f"{ticker} RSI",
                line=dict(color=rsi_color, width=2),
                showlegend=False,
            ),
            row=row,
            col=col,
        )

        fig.add_trace(
            go.Scatter(
                x=x_labels,
                y=df_plot["RSI_MA"],
                mode="lines",
                name=f"{ticker} SMA",
                line=dict(color="#f59e0b", width=1.5, dash="dot"),
                showlegend=False,
            ),
            row=row,
            col=col,
        )

        fig.add_hline(y=70, line_dash="dash", line_color="rgba(255,255,255,0.2)", row=row, col=col)
        fig.add_hline(y=30, line_dash="dash", line_color="rgba(255,255,255,0.2)", row=row, col=col)

        if ticker in INVERTED_TICKERS:
            fig.update_yaxes(range=[90, 10], row=row, col=col)
        else:
            fig.update_yaxes(range=[10, 90], row=row, col=col)

        fig.update_xaxes(nticks=6, row=row, col=col)

    fig.update_layout(
        height=240 * num_rows,
        template="plotly_dark",
        margin=dict(l=20, r=20, t=40, b=20),
    )

    st.plotly_chart(fig, use_container_width=True)


def main() -> None:
    st.title("VIÉS MACRO 15M")

    with st.sidebar:
        st.header("Parâmetros")
        rsi_len = st.number_input("Período RSI", min_value=2, max_value=50, value=RSI_PERIOD)
        ma_len = st.number_input("Média Móvel no RSI (SMA)", min_value=2, max_value=50, value=RSI_MA_PERIOD)
        include_extended = st.checkbox("Incluir Pré / After-Market", value=False)
        auto_refresh = st.checkbox("Auto-atualizar (30s)", value=True)
        candles_chart = st.slider("Candles visíveis no gráfico", min_value=30, max_value=150, value=78)

        if st.button("Atualizar Agora", use_container_width=True):
            st.cache_data.clear()

    if auto_refresh:
        @st.fragment(run_every=30)
        def auto_updating_view():
            render_dashboard(int(rsi_len), int(ma_len), include_extended, candles_chart)

        auto_updating_view()
    else:
        render_dashboard(int(rsi_len), int(ma_len), include_extended, candles_chart)


if __name__ == "__main__":
    main()
