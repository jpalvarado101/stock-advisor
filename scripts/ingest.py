import yfinance as yf
import pandas as pd
from datetime import date


LOOKBACK_YEARS = 6


def fetch_one(symbol: str) -> pd.DataFrame:
    df = yf.download(symbol, period=f"{LOOKBACK_YEARS}y", interval="1d", auto_adjust=False)
    df = df.rename(columns=str.lower).reset_index().rename(columns={"index": "date", "adj close": "adj_close"})
    return df