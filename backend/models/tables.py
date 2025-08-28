# backend/models/tables.py
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy import String, Float, Integer, Date, DateTime, ForeignKey, UniqueConstraint, Index


class Base(DeclarativeBase):
    pass


class Ticker(Base):
    __tablename__ = "tickers"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    symbol: Mapped[str] = mapped_column(String(16), unique=True, index=True)
    sector: Mapped[str | None] = mapped_column(String(64))


class Price(Base):
    __tablename__ = "prices"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    ticker_id: Mapped[int] = mapped_column(ForeignKey("tickers.id"), index=True)
    date: Mapped[Date] = mapped_column(Date, index=True)
    open: Mapped[float] = mapped_column(Float)
    high: Mapped[float] = mapped_column(Float)
    low: Mapped[float] = mapped_column(Float)
    close: Mapped[float] = mapped_column(Float)
    adj_close: Mapped[float] = mapped_column(Float)
    volume: Mapped[float] = mapped_column(Float)
    __table_args__ = (UniqueConstraint("ticker_id", "date", name="ux_price_ticker_date"),)


class FeatureRow(Base):
    __tablename__ = "features"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    ticker_id: Mapped[int] = mapped_column(ForeignKey("tickers.id"), index=True)
    date: Mapped[Date] = mapped_column(Date, index=True)
    r1: Mapped[float] = mapped_column(Float) # log-return t-0→t
    rv_10: Mapped[float] = mapped_column(Float) # realized vol 10d
    z_30: Mapped[float] = mapped_column(Float) # price z-score 30d
    rsi_14: Mapped[float] = mapped_column(Float)
    macd: Mapped[float] = mapped_column(Float)
    macd_sig: Mapped[float] = mapped_column(Float)
    sma_50: Mapped[float] = mapped_column(Float)
    sma_200: Mapped[float] = mapped_column(Float)
    vol_spike: Mapped[float] = mapped_column(Float)
    gbm_mu: Mapped[float] = mapped_column(Float) # μ̂ from window
    gbm_sigma: Mapped[float] = mapped_column(Float) # σ̂ from window
    __table_args__ = (UniqueConstraint("ticker_id", "date", name="ux_feat_ticker_date"),)


class ModelRun(Base):
    __tablename__ = "model_runs"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    created_at: Mapped[DateTime]
    algo: Mapped[str] = mapped_column(String(64)) # "mlp" | "gru"
    lookback: Mapped[int] = mapped_column(Integer)
    horizon: Mapped[int] = mapped_column(Integer) # 1-day ahead
    notes: Mapped[str | None] = mapped_column(String(256))


class Prediction(Base):
    __tablename__ = "predictions"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    run_id: Mapped[int] = mapped_column(ForeignKey("model_runs.id"), index=True)
    ticker_id: Mapped[int] = mapped_column(ForeignKey("tickers.id"), index=True)
    date: Mapped[Date] = mapped_column(Date, index=True)
    mu: Mapped[float] = mapped_column(Float) # predicted mean log‑return
    var: Mapped[float] = mapped_column(Float) # predicted variance
    y_true: Mapped[float | None] = mapped_column(Float) # filled after t+1
    __table_args__ = (UniqueConstraint("run_id", "ticker_id", "date", name="ux_pred_run_ticker_date"),)