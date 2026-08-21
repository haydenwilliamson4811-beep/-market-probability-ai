
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss, accuracy_score
from sklearn.pipeline import Pipeline

st.set_page_config(
    page_title="Market Probability AI",
    page_icon="📈",
    layout="wide",
)

st.title("📈 Market Probability AI")
st.caption("Research dashboard — estimates probabilities from historical market data. Not financial advice.")

FEATURES = [
    "ret_1", "ret_2", "ret_5", "ret_10", "ret_20",
    "ma_5_dist", "ma_10_dist", "ma_20_dist", "ma_50_dist", "ma_200_dist",
    "vol_10", "vol_20", "range_1", "gap_1",
    "rsi_14", "volume_z_20",
    "market_ret_1", "market_ret_5", "market_vol_20",
]

DEFAULT_TICKERS = "AAPL, MSFT, NVDA, AMZN, META, GOOGL, TSLA, JPM, XOM, COST"

def download_one(ticker: str, start: str) -> pd.DataFrame:
    df = yf.download(
        ticker,
        start=start,
        auto_adjust=True,
        progress=False,
        actions=False,
        threads=False,
    )
    if df.empty:
        raise ValueError(f"No data returned for {ticker}")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.index = pd.to_datetime(df.index).tz_localize(None)
    return df[["Open", "High", "Low", "Close", "Volume"]].copy()

def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def add_features(price, market, ticker, horizon, target_return, include_target=True):
    df = price.copy()
    close = df["Close"]

    for n in [1, 2, 5, 10, 20]:
        df[f"ret_{n}"] = close.pct_change(n)

    for n in [5, 10, 20, 50, 200]:
        ma = close.rolling(n).mean()
        df[f"ma_{n}_dist"] = close / ma - 1

    ret = close.pct_change()
    df["vol_10"] = ret.rolling(10).std()
    df["vol_20"] = ret.rolling(20).std()
    df["range_1"] = (df["High"] - df["Low"]) / close
    df["gap_1"] = df["Open"] / close.shift(1) - 1
    df["rsi_14"] = rsi(close)
    volume_mean = df["Volume"].rolling(20).mean()
    volume_std = df["Volume"].rolling(20).std()
    df["volume_z_20"] = (df["Volume"] - volume_mean) / volume_std.replace(0, np.nan)

    m = market.copy()
    m["market_ret_1"] = m["Close"].pct_change()
    m["market_ret_5"] = m["Close"].pct_change(5)
    m["market_vol_20"] = m["Close"].pct_change().rolling(20).std()
    df = df.join(m[["market_ret_1", "market_ret_5", "market_vol_20"]], how="left")

    df["ticker"] = ticker

    if include_target:
        df["future_return"] = close.shift(-horizon) / close - 1
        df["target"] = (df["future_return"] >= target_return).astype(float)
        df.loc[df["future_return"].isna(), "target"] = np.nan

    return df

def make_model():
    prep = ColumnTransformer(
        [("num", SimpleImputer(strategy="median"), FEATURES)],
        remainder="drop",
    )
    clf = HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_iter=250,
        max_leaf_nodes=15,
        min_samples_leaf=40,
        l2_regularization=1.0,
        random_state=42,
    )
    return Pipeline([("prep", prep), ("model", clf)])

def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p)).reshape(-1, 1)

def calibrated_probs(base, calibrator, X):
    raw = base.predict_proba(X[FEATURES])[:, 1]
    return calibrator.predict_proba(logit(raw))[:, 1]

with st.sidebar:
    st.header("Model settings")
    ticker_text = st.text_area("Tickers", DEFAULT_TICKERS, height=120)
    market_ticker = st.text_input("Market benchmark", "SPY")
    start_date = st.text_input("History start date", "2014-01-01")
    horizon = st.slider("Prediction horizon (trading days)", 1, 20, 5)
    target_percent = st.slider("Target return", 0.25, 10.0, 1.0, 0.25)
    signal_threshold = st.slider("Signal threshold", 50, 90, 65, 1)
    run_button = st.button("Train & Scan", type="primary", use_container_width=True)

st.info(
    f"Current target: probability the stock closes at least "
    f"**{target_percent:.2f}% higher in {horizon} trading days**."
)

if run_button:
    tickers = [t.strip().upper() for t in ticker_text.split(",") if t.strip()]
    target_return = target_percent / 100
    threshold = signal_threshold / 100

    progress = st.progress(0, text="Downloading benchmark data...")
    try:
        market = download_one(market_ticker.upper(), start_date)
    except Exception as exc:
        st.error(f"Could not load benchmark {market_ticker}: {exc}")
        st.stop()

    raw = {}
    frames = []
    errors = []

    for i, ticker in enumerate(tickers, start=1):
        progress.progress(i / max(len(tickers), 1), text=f"Loading {ticker}...")
        try:
            px = download_one(ticker, start_date)
            raw[ticker] = px
            feat = add_features(px, market, ticker, horizon, target_return, True)
            frames.append(feat)
        except Exception as exc:
            errors.append(f"{ticker}: {exc}")

    if not frames:
        st.error("No usable ticker data was downloaded.")
        st.stop()

    data = pd.concat(frames).sort_index()
    data = data.dropna(subset=FEATURES + ["target", "future_return"])
    data["target"] = data["target"].astype(int)

    dates = np.array(sorted(data.index.unique()))
    if len(dates) < 500:
        st.error("Not enough historical data. Use an earlier start date.")
        st.stop()

    train_end = dates[int(len(dates) * 0.70)]
    calib_end = dates[int(len(dates) * 0.85)]

    train = data[data.index < train_end]
    calib = data[(data.index >= train_end) & (data.index < calib_end)]
    test = data[data.index >= calib_end].copy()

    progress.progress(0.88, text="Training model...")
    base = make_model()
    base.fit(train[FEATURES], train["target"])

    raw_calib = base.predict_proba(calib[FEATURES])[:, 1]
    calibrator = LogisticRegression(random_state=42)
    calibrator.fit(logit(raw_calib), calib["target"])

    test["prob"] = calibrated_probs(base, calibrator, test)
    test["pred"] = (test["prob"] >= 0.5).astype(int)

    auc = roc_auc_score(test["target"], test["prob"])
    brier = brier_score_loss(test["target"], test["prob"])
    accuracy = accuracy_score(test["target"], test["pred"])

    signals = test[test["prob"] >= threshold]
    hit_rate = signals["target"].mean() if len(signals) else np.nan
    avg_return = signals["future_return"].mean() if len(signals) else np.nan

    progress.progress(0.96, text="Scanning latest market data...")
    scan_rows = []
    for ticker, px in raw.items():
        feat = add_features(px, market, ticker, horizon, target_return, False)
        feat = feat.dropna(subset=FEATURES)
        if feat.empty:
            continue
        latest = feat.iloc[[-1]]
        prob = float(calibrated_probs(base, calibrator, latest)[0])
        scan_rows.append({
            "Ticker": ticker,
            "Date": latest.index[0].date().isoformat(),
            "Close": float(latest["Close"].iloc[0]),
            "Probability": prob,
            "Signal": "BUY-CANDIDATE" if prob >= threshold else "NO SIGNAL",
            "RSI 14": float(latest["rsi_14"].iloc[0]),
            "20D Volatility": float(latest["vol_20"].iloc[0]),
        })

    scan = pd.DataFrame(scan_rows).sort_values("Probability", ascending=False)
    progress.progress(1.0, text="Done")

    st.subheader("Out-of-sample model test")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("ROC AUC", f"{auc:.3f}")
    c2.metric("Brier score", f"{brier:.3f}")
    c3.metric("Accuracy", f"{accuracy:.1%}")
    c4.metric(f"Signals ≥ {signal_threshold}%", f"{len(signals):,}")
    c5.metric("Signal hit rate", "N/A" if np.isnan(hit_rate) else f"{hit_rate:.1%}")

    if not np.isnan(avg_return):
        st.caption(f"Average {horizon}-day return among test signals: {avg_return:.2%}")

    st.subheader("Latest scanner")
    display = scan.copy()
    display["Probability"] = display["Probability"].map(lambda x: f"{x:.1%}")
    display["Close"] = display["Close"].map(lambda x: f"{x:,.2f}")
    display["RSI 14"] = display["RSI 14"].map(lambda x: f"{x:.1f}")
    display["20D Volatility"] = display["20D Volatility"].map(lambda x: f"{x:.2%}")
    st.dataframe(display, use_container_width=True, hide_index=True)

    st.download_button(
        "Download latest scan CSV",
        data=scan.to_csv(index=False).encode("utf-8"),
        file_name="latest_stock_probability_scan.csv",
        mime="text/csv",
    )

    st.subheader("Probability calibration")
    buckets = pd.cut(
        test["prob"],
        bins=[0, .40, .50, .60, .70, .80, .90, 1.0],
        include_lowest=True,
    )
    calibration = (
        test.assign(bucket=buckets)
        .groupby("bucket", observed=False)
        .agg(
            Predictions=("target", "size"),
            Average_predicted_probability=("prob", "mean"),
            Actual_success_rate=("target", "mean"),
        )
        .reset_index()
    )
    st.dataframe(calibration, use_container_width=True, hide_index=True)

    st.subheader("Probability distribution")
    st.bar_chart(test["prob"].value_counts(bins=20).sort_index())

    if errors:
        with st.expander("Ticker download warnings"):
            for err in errors:
                st.write(err)

    st.warning(
        "A high model probability is not a guarantee of profit. Before real-money use, "
        "add walk-forward testing, realistic spreads/fees, portfolio sizing and paper trading."
    )
else:
    st.markdown(
        """
### How to use it
1. Enter the stocks you want to analyse.
2. Pick the prediction horizon and required return.
3. Choose a probability threshold.
4. Press **Train & Scan**.
5. Compare the model's predicted probabilities with its actual out-of-sample success rates.

For ASX stocks, use Yahoo Finance symbols such as `BHP.AX`, `CBA.AX`, `FMG.AX`.
"""
    )
