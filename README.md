# Stock Probability AI — MVP

This is a research/backtesting starter system. It estimates the probability that
a stock's adjusted close will be at least a chosen percentage higher after a
chosen number of trading days.

Default target:
- Horizon: 5 trading days
- Required return: +1%
- Signal threshold: 65%

## Why this design

The system deliberately separates history into three chronological blocks:

1. **Training data** — fits the machine-learning model.
2. **Calibration data** — converts raw model scores into more realistic probabilities.
3. **Test data** — untouched until final evaluation.

That is much safer than randomly shuffling stock-market rows, which can create
look-ahead leakage.

## Install

Python 3.11+ is recommended.

```bash
python -m venv .venv

# macOS/Linux
source .venv/bin/activate

# Windows PowerShell
.venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

## Run

```bash
python market_ai.py
```

The script will:
- download historical adjusted daily prices;
- calculate price, trend, momentum, volatility and volume features;
- train a nonlinear classifier;
- calibrate its probabilities on a later period;
- evaluate on a still-later unseen test period;
- print a calibration table;
- scan the latest data;
- save predictions to CSV.

## Configure

Edit `config.py`.

Useful settings:
- `TICKERS`
- `HORIZON`
- `TARGET_RETURN`
- `SIGNAL_THRESHOLD`
- `ROUND_TRIP_COST`

For ASX shares, Yahoo Finance normally uses `.AX`, for example `BHP.AX`,
`CBA.AX`, and an Australian benchmark such as `^AXJO` can be used as the market
ticker.

## How to judge whether the probability is useful

Do **not** judge the model by one winning trade.

A probability model is useful only if its predictions remain calibrated out of
sample. As an example, among a large number of predictions around 70%, the
event should happen roughly 70% of the time.

Also examine:
- number of predictions in each probability bucket;
- Brier score;
- ROC AUC;
- hit rate only on high-probability signals;
- returns after realistic fees/slippage;
- performance through different market regimes.

## Important limitations

- `yfinance` is convenient for prototyping, not an institutional execution feed.
- Daily OHLCV alone may not contain enough predictive information for a durable edge.
- The current backtest treats signals as observations, not a full portfolio with
  capital constraints, overlapping positions, spreads, taxes and execution.
- Historical performance does not guarantee future returns.
- Do not connect this directly to real-money auto-trading before extensive
  walk-forward and paper-trading validation.

## Strong next upgrades

1. Proper walk-forward portfolio backtest.
2. Larger liquid-stock universe.
3. Earnings/fundamental features.
4. News/sentiment features.
5. Relative strength and sector features.
6. Regime detection.
7. Broker/data-provider API.
8. Paper-trading engine.
9. Dashboard and alerts.
