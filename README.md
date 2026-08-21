# Market Probability AI App

A Streamlit app that trains a machine-learning model on historical stock data,
calibrates its probabilities, tests it chronologically on unseen data, and scans
the latest available prices.

## Start the app

Install Python 3.11+.

### Windows

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

### macOS / Linux

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Your browser should open automatically.

## What the probability means

If your settings are:
- target return = 1%
- horizon = 5 days

then a displayed probability of 70% means:

> Based on patterns learned from the historical dataset, the calibrated model
> estimates a 70% chance the adjusted closing price will be at least 1% higher
> five trading days later.

It does **not** mean there is a guaranteed 70% chance of making money in a live
trade.

## Included

- Historical price/volume download
- Momentum features
- Moving-average features
- RSI
- Volatility
- Volume anomaly
- Benchmark-market features
- Chronological train/calibration/test split
- Nonlinear machine-learning classifier
- Probability calibration
- Out-of-sample AUC, Brier score and accuracy
- High-probability signal hit rate
- Latest stock scanner
- CSV export

## Recommended next upgrades

- Walk-forward retraining
- Realistic trading costs and slippage
- Stop-loss / take-profit outcome modeling
- Portfolio sizing
- Fundamentals
- Earnings data
- News sentiment
- Options-market features
- Sector relative strength
- Paper-trading broker integration
- User authentication and cloud deployment

This software is for research and education and is not financial advice.
