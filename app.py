# Universo
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
