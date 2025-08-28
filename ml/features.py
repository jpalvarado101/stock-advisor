import numpy as np, pandas as pd


def logret(s: pd.Series) -> pd.Series:
    return np.log(s).diff()


def realized_vol(r: pd.Series, w: int) -> pd.Series:
    return r.rolling(w).std() * np.sqrt(252)


# MLE for GBM on log-returns r_t ~ N((μ − 0.5σ²)Δt, σ²Δt)
# For Δt=1/252, we estimate μ, σ on a rolling window


def gbm_params(r: pd.Series, w: int, dt: float=1/252):
    m = r.rolling(w).mean()
    s = r.rolling(w).std(ddof=0)
    sigma = s / np.sqrt(dt)
    mu = (m / dt) + 0.5 * sigma**2
    return mu, sigma


# Indicators


def rsi(series: pd.Series, period: int=14):
    diff = series.diff()
    up = diff.clip(lower=0).ewm(alpha=1/period, adjust=False).mean()
    down = (-diff.clip(upper=0)).ewm(alpha=1/period, adjust=False).mean()
    rs = up / down
    return 100 - (100 / (1 + rs))


def macd(series: pd.Series, fast=12, slow=26, signal=9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    sig = macd.ewm(span=signal, adjust=False).mean()
    return macd, sig


def make_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["r1"] = logret(out["adj_close"]).fillna(0)
    out["rv_10"] = realized_vol(out["r1"], 10)
    out["z_30"] = (out["adj_close"] - out["adj_close"].rolling(30).mean()) / out["adj_close"].rolling(30).std()
    out["sma_50"] = out["adj_close"].rolling(50).mean()
    out["sma_200"] = out["adj_close"].rolling(200).mean()
    out["rsi_14"] = rsi(out["adj_close"])
    out["macd"], out["macd_sig"] = macd(out["adj_close"])
    out["vol_spike"] = out["volume"] / out["volume"].rolling(30).mean()
    mu, sigma = gbm_params(out["r1"], 60) # 60d rolling GBM params
    out["gbm_mu"], out["gbm_sigma"] = mu, sigma
    return out.dropna()