CREATE TABLE tickers (
    id serial PRIMARY KEY,
    symbol TEXT UNIQUE NOT NULL,
    name TEXT,
    exchange TEXT, 
    sector TEXT,
    industry TEXT,
    category TEXT,
    created_at TIMESTAMPTZ DEFAULT now()
)

CREATE TABLE prices (
    ticker_id INT REFERENCES tickers(id),
    ts DATE NOT NULL,
    open NUMERIC(18,6), high NUMERIC(18,6), low NUMERIC(18,6), close NUMERIC(18,6),
    volume BIGINT,
    adj_close NUMERIC(18,6),
    PRIMARY KEY (ticker_id, ts)
)

CREATE TABLE fundamentals (
  ticker_id INT REFERENCES tickers(id),
  period DATE NOT NULL,
  revenue NUMERIC, eps NUMERIC, pe NUMERIC, pb NUMERIC, debt_to_equity NUMERIC,
  free_cash_flow NUMERIC,
  PRIMARY KEY (ticker_id, period)
);

CREATE TABLE sentiments (
  id BIGSERIAL PRIMARY KEY,
  ticker_id INT REFERENCES tickers(id),
  source TEXT,
  posted_at TIMESTAMPTZ,
  polarity NUMERIC,
  subjectivity NUMERIC,
  volume_weight NUMERIC,
  metadata JSONB
);

CREATE TABLE features (
  ticker_id INT REFERENCES tickers(id),
  ts DATE NOT NULL,
  feature_vector REAL[],
  label SMALLINT,
  PRIMARY KEY (ticker_id, ts)
);

CREATE TABLE backtests (
  id UUID PRIMARY KEY,
  strategy TEXT,
  params JSONB,
  start_date DATE, end_date DATE,
  created_at TIMESTAMPTZ DEFAULT now()
);
