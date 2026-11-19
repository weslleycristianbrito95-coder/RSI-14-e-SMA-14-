import time
from datetime import datetime
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pytz
import streamlit as st
import streamlit.components.v1 as components
import yfinance as yf

st.set_page_config(
    page_title="Scanner Intraday RSI vs SMA",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Lista de ativos organizada por segmento de mercado
TICKERS = [
    # --- ÍNDICES & ETFS DE SETORES ---
    "QQQ",       # Nasdaq 100
    "SOX",       # PHLX Semiconductor Index
    "SMH",       # VanEck Semiconductor ETF
    "IGV",       # iShares Expanded Tech-Software ETF
    "XLK",       # Technology Select Sector SPDR
    "XLC",       # Communication Services Select Sector
    "XLY",       # Consumer Discretionary Select Sector
    "XLF",       # Financial Select Sector SPDR

    # --- AÇÕES LÍDERES (TECNOLOGIA / CHIPS) ---
    "AAPL",      # Apple
    "NVDA",      # NVIDIA
    "MSFT",      # Microsoft
    "AMZN",      # Amazon
    "MU",        # Micron Technology
    "AMD",       # Advanced Micro Devices
    "TSLA",      # Tesla

    # --- PARES DE MOEDAS (FOREX) ---
    "EURUSD=X",  # Euro / Dólar Americano
    "GBPUSD=X",  # Libra Esterlina / Dólar Americano
    "USDJPY=X",  # Dólar Americano / Iene Japonês
    "AUDUSD=X",  # Dólar Australiano / Dólar Americano
    "NZDUSD=X",  # Dólar Neozelandês / Dólar Americano
    "AUDJPY=X",  # Dólar Australiano / Iene Japonês
    "NZDJPY=X",  # Dólar Neozelandês / Iene Japonês

    # --- CRIPTOMOEDAS ---
    "BTC-USD",   # Bitcoin

    # --- VOLATILIDADE ---
    "^VIX",      # Índice de Volatilidade CBOE

    # --- MACRO & RENDA FIXA ---
    "DX-Y.NYB",  # Índice Dólar (DXY)
    "US02Y=F"    # Rendimento de Títulos EUA 2 anos (Yield)
]

RSI_PERIOD = 14
RSI_MA_PERIOD = 20
TIMEZONE_BRT = pytz.timezone("America/Sao_Paulo")

# Ativos que operam com a lógica e escala invertidas
INVERTED_TICKERS = ["^VIX", "DX-Y.NYB"]

# Galeria de sons disponíveis para o alarme
SOUND_GALLERY = {
    "Sino Clássico": "https://raw.githubusercontent.com/carlesb/react-native-sound-playerview/master/assets/bell.mp3",
    "Sirene de Emergência": "https://www.soundjay.com/buttons/sounds/button-10.mp3",
    "Acorde de Piano": "https://www.soundjay.com/misc/sounds/piano-g-major.mp3",
    "Bip Eletrônico": "https://www.soundjay.com/buttons/sounds/button-3.mp3"
}


def calculate_rsi_wilder(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50.0)


@st.cache_data(ttl=55, show_spinner=False)
def fetch_and_process_data(
    tickers: Tuple[str, ...],
    rsi_len: int,
    ma_len: int,
    include_extended: bool,
) -> Dict[str, pd.DataFrame]:
    processed: Dict[str, pd.DataFrame] = {}

    try:
        raw = yf.download(
            tickers=list(tickers),
            period="5d",
            interval="15m",
            group_by="ticker",
            prepost=include_extended,
            threads=True,
            progress=False,
        )
    except Exception as exc:
        st.error(f"Erro na conexão com Yahoo Finance: {exc}")
        return processed

    if raw is None or raw.empty:
        return processed

    for ticker in tickers:
        try:
            if len(tickers) > 1:
                if ticker not in raw.columns.get_level_values(0):
                    continue
                df = raw[ticker].copy()
            else:
                df = raw.copy()

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

            processed[ticker] = df
        except Exception as exc:
            st.warning(f"Erro ao processar {ticker}: {exc}")

    return processed


def play_alarm_sound(sound_url: str) -> None:
    sound_html = f"""
    <div>
        <audio id="alarmAudio" autoplay>
            <source src="{sound_url}" type="audio/mp3">
        </audio>
        <script>
        (function() {{
            const audioEl = document.getElementById("alarmAudio");
            if (audioEl) {{
                audioEl.play().catch(function() {{
                    try {{
                        const AudioCtx = window.AudioContext || window.webkitAudioContext;
                        const ctx = new AudioCtx();
                        const osc = ctx.createOscillator();
                        const gain = ctx.createGain();
                        osc.type = 'sine';
                        osc.frequency.setValueAtTime(880, ctx.currentTime);
                        gain.gain.setValueAtTime(0.25, ctx.currentTime);
                        gain.gain.exponentialRampToValueAtTime(0.01, ctx.currentTime + 0.35);
                        osc.connect(gain);
                        gain.connect(ctx.destination);
                        osc.start();
                        osc.stop(ctx.currentTime + 0.38);
                    }} catch (e) {{}}
                }});
            }}
        }})();
        </script>
    </div>
    """
    components.html(sound_html, height=0, width=0)


def main() -> None:
    st.title("VIÉS MACRO 15M")

    with st.sidebar:
        st.header("Parâmetros")
        rsi_len = st.number_input("Período RSI", min_value=2, max_value=50, value=RSI_PERIOD)
        ma_len = st.number_input("Média Móvel no RSI (SMA)", min_value=2, max_value=50, value=RSI_MA_PERIOD)
        include_extended = st.checkbox("Incluir Pré / After-Market", value=False)
        auto_refresh = st.checkbox("Auto-atualizar (60s)", value=True)
        candles_chart = st.slider("Candles visíveis no gráfico", min_value=30, max_value=150, value=78)

        st.header("Configuração de Alerta")
        enable_sound = st.checkbox("Ativar Alerta Sonoro", value=True)
        selected_sound_name = st.selectbox("Escolher Som do Alarme", list(SOUND_GALLERY.keys()))
        alert_percentage = st.slider(
            "Gatilho do Alerta (% do mesmo lado)", min_value=50, max_value=100, value=90, step=1
        )
        if st.button("Testar Som Selecionado", use_container_width=True):
            play_alarm_sound(SOUND_GALLERY[selected_sound_name])

        st.divider()
        if st.button("Atualizar Agora", use_container_width=True):
            st.cache_data.clear()
            st.rerun()

    with st.spinner("Sincronizando cotações (15m)..."):
        data_map = fetch_and_process_data(
            tuple(TICKERS), int(rsi_len), int(ma_len), include_extended
        )

    if not data_map:
        st.error("Não foi possível obter dados do Yahoo Finance no momento. Tente novamente em instantes.")
        return

    summary_rows = []
    acima_count = 0

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

        if display_above and not display_was_above:
            evento = "CRUZOU PARA CIMA"
        elif not display_above and display_was_above:
            evento = "CRUZOU PARA BAIXO"
        else:
            evento = "Mantendo tendência"

        is_four_decimals = "=X" in ticker or "DX" in ticker or "US" in ticker

        summary_rows.append(
            {
                "Ativo": ticker,
                "Último Candle (BRT)": df.index[-1].strftime("%d/%m %H:%M"),
                "Preço": round(float(last["Close"]), 4) if is_four_decimals else round(float(last["Close"]), 2),
                rsi_col_name: round(float(last["RSI"]), 2),
                ma_col_name: round(float(last["RSI_MA"]), 2),
                "Spread (RSI - MA)": round(float(last["Spread"]), 2),
                "Status": "ACIMA DA MÉDIA" if display_above else "ABAIXO DA MÉDIA",
                "Sinal Último Candle": evento,
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    total_ativos = len(summary_df)

    if total_ativos > 0:
        pct_acima = acima_count / total_ativos
        pct_abaixo = (total_ativos - acima_count) / total_ativos

        target_ratio = alert_percentage / 100.0

        if pct_acima >= target_ratio or pct_abaixo >= target_ratio:
            if enable_sound:
                sound_url = SOUND_GALLERY[selected_sound_name]
                play_alarm_sound(sound_url)
            st.toast(f"⚠️ ALERTA: Mais de {alert_percentage}% dos ativos estão no mesmo sentido!", icon="🚨")

    now_brt = datetime.now(TIMEZONE_BRT).strftime("%H:%M:%S")
    col1, col2, col3 = st.columns(3)
    col1.metric("Ativos Acima da Média (RSI > SMA)", f"{acima_count} de {total_ativos}")
    col2.metric("Ativos Abaixo da Média (RSI < SMA)", f"{total_ativos - acima_count} de {total_ativos}")
    col3.metric("Última Atualização (Brasília)", now_brt)

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

    formatted_prices = []
    for _, row in summary_df.iterrows():
        ativo_str = str(row["Ativo"])
        if "=X" in ativo_str or "DX" in ativo_str or "US" in ativo_str:
            formatted_prices.append(f"{row['Preço']:.4f}")
        else:
            formatted_prices.append(f"${row['Preço']:.2f}")
    summary_df["Preço Formated"] = formatted_prices

    cols_order = [
        "Ativo", "Último Candle (BRT)", "Preço Formated", rsi_col_name,
        ma_col_name, "Spread (RSI - MA)", "Status", "Sinal Último Candle"
    ]
    display_df = summary_df[cols_order].rename(columns={"Preço Formated": "Preço"})

    styled_df = (
        display_df.style
        .map(highlight_status, subset=["Status"])
        .map(highlight_event, subset=["Sinal Último Candle"])
        .format({
            rsi_col_name: "{:.2f}",
            ma_col_name: "{:.2f}",
            "Spread (RSI - MA)": "{:+.2f}",
        })
    )

    st.dataframe(styled_df, use_container_width=True, hide_index=True)

    st.subheader(f"Curva RSI({rsi_len}) vs. Média Móvel ({ma_len}) — Gráfico de 15 Minutos")

    active_tickers = [t for t in TICKERS if t in data_map]
    num_plots = len(active_tickers)
    num_cols = 2
    num_rows = (num_plots + 1) // num_cols

    fig = make_subplots(
        rows=num_rows,
        cols=num_cols,
        subplot_titles=[
            f"{t} — {'ABAIXO' if data_map[t]['Acima_Media'].iloc[-1] else 'ACIMA'} (Invertido)" if t in INVERTED_TICKERS
            else f"{t} — {'ACIMA' if data_map[t]['Acima_Media'].iloc[-1] else 'ABAIXO'}"
            for t in active_tickers
        ],
        vertical_spacing=0.04,
        horizontal_spacing=0.06,
    )

    idx = 0
    for ticker in active_tickers:
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

        idx += 1

    fig.update_layout(
        height=240 * num_rows,
        template="plotly_dark",
        margin=dict(l=20, r=20, t=40, b=20),
    )

    st.plotly_chart(fig, use_container_width=True)

    if auto_refresh:
        time.sleep(60)
        st.rerun()


if __name__ == "__main__":
    main()
