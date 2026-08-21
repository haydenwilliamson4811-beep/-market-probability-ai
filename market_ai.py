\
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yfinance as yf

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from config import (
    TICKERS,
    MARKET_TICKER,
    HORIZON,
    TARGET_RETURN,
    ROUND_TRIP_COST,
    SIGNAL_THRESHOLD,
    START_DATE,
    RANDOM_STATE,
)

MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

FEATURES = [
    "ret_1", "ret_2", "ret_5", "ret_10", "ret_20",
    "ma_5_dist", "ma_10_dist", "ma_20_dist", "ma_50_dist", "ma_200_dist",
    "vol_10", "vol_20", "range_1", "gap_1",
    "rsi_14", "volume_z_20",
    "market_ret_1", "market_ret_5", "market_vol_20",
]


@dataclass
class SplitData:
    train: pd.DataFrame
    calib: pd.DataFrame
    test: pd.DataFrame


def _download_one(ticker: str, start: str) -> pd.DataFrame:
    df = yf.download(
        ticker,
        start=start,
        auto_adjust=True,
        progress=False,
        actions=False,
        threads=False,
    )
    if df.empty:
        raise RuntimeError(f"No data returned for {ticker}")

    # yfinance can return MultiIndex columns for some versions/settings.
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    wanted = ["Open", "High", "Low", "Close", "Volume"]
    missing = [c for c in wanted if c not in df.columns]
    if missing:
        raise RuntimeError(f"{ticker}: missing columns: {missing}")

    out = df[wanted].copy()
    out.index = pd.to_datetime(out.index).tz_localize(None)
    return out


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def add_features(
    price: pd.DataFrame,
    market: pd.DataFrame,
    ticker: str,
    include_target: bool = True,
) -> pd.DataFrame:
    df = price.copy()
    close = df["Close"]

    for n in [1, 2, 5, 10, 20]:
        df[f"ret_{n}"] = close.pct_change(n)

    for n in [5, 10, 20, 50, 200]:
        ma = close.rolling(n).mean()
        df[f"ma_{n}_dist"] = close / ma - 1

    daily_ret = close.pct_change()
    df["vol_10"] = daily_ret.rolling(10).std()
    df["vol_20"] = daily_ret.rolling(20).std()
    df["range_1"] = (df["High"] - df["Low"]) / close
    df["gap_1"] = df["Open"] / close.shift(1) - 1
    df["rsi_14"] = rsi(close, 14)

    vol_mean = df["Volume"].rolling(20).mean()
    vol_std = df["Volume"].rolling(20).std()
    df["volume_z_20"] = (df["Volume"] - vol_mean) / vol_std.replace(0, np.nan)

    m = market.copy()
    m["market_ret_1"] = m["Close"].pct_change(1)
    m["market_ret_5"] = m["Close"].pct_change(5)
    m["market_vol_20"] = m["Close"].pct_change().rolling(20).std()
    df = df.join(m[["market_ret_1", "market_ret_5", "market_vol_20"]], how="left")

    df["ticker"] = ticker

    if include_target:
        df["future_return"] = close.shift(-HORIZON) / close - 1
        df["target"] = (df["future_return"] >= TARGET_RETURN).astype(float)
        # Last HORIZON rows do not yet have a known outcome.
        df.loc[df["future_return"].isna(), "target"] = np.nan

    return df


def build_dataset() -> tuple[pd.DataFrame, dict[str, pd.DataFrame], pd.DataFrame]:
    print("Downloading market data...")
    market = _download_one(MARKET_TICKER, START_DATE)
    raw = {}
    frames = []

    for ticker in TICKERS:
        print(f"  {ticker}")
        try:
            px = _download_one(ticker, START_DATE)
            raw[ticker] = px
            feat = add_features(px, market, ticker, include_target=True)
            frames.append(feat)
        except Exception as exc:
            print(f"  WARNING: skipped {ticker}: {exc}")

    if not frames:
        raise RuntimeError("No ticker data could be downloaded.")

    data = pd.concat(frames).sort_index()
    data = data.dropna(subset=FEATURES + ["target", "future_return"])
    data["target"] = data["target"].astype(int)
    return data, raw, market


def chronological_split(data: pd.DataFrame) -> SplitData:
    # Split by unique dates so rows from the same day never straddle splits.
    dates = np.array(sorted(data.index.unique()))
    if len(dates) < 500:
        raise RuntimeError("Not enough history after feature generation.")

    train_end = dates[int(len(dates) * 0.70)]
    calib_end = dates[int(len(dates) * 0.85)]

    train = data[data.index < train_end].copy()
    calib = data[(data.index >= train_end) & (data.index < calib_end)].copy()
    test = data[data.index >= calib_end].copy()

    return SplitData(train=train, calib=calib, test=test)


def make_base_model() -> Pipeline:
    # Tree model handles nonlinear relationships. Median imputation is defensive;
    # features should normally already be non-null.
    pre = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                ]),
                FEATURES,
            )
        ],
        remainder="drop",
    )

    clf = HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_iter=250,
        max_leaf_nodes=15,
        min_samples_leaf=40,
        l2_regularization=1.0,
        random_state=RANDOM_STATE,
    )

    return Pipeline([("pre", pre), ("model", clf)])


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p)).reshape(-1, 1)


def fit_models(split: SplitData):
    base = make_base_model()
    base.fit(split.train[FEATURES], split.train["target"])

    # Chronological probability calibration:
    # base model sees TRAIN only; calibrator sees CALIBRATION only.
    raw_calib = base.predict_proba(split.calib[FEATURES])[:, 1]
    calibrator = LogisticRegression(random_state=RANDOM_STATE)
    calibrator.fit(_logit(raw_calib), split.calib["target"])

    return base, calibrator


def predict_calibrated(base, calibrator, X: pd.DataFrame) -> np.ndarray:
    raw = base.predict_proba(X[FEATURES])[:, 1]
    return calibrator.predict_proba(_logit(raw))[:, 1]


def calibration_table(y_true: pd.Series, p: np.ndarray) -> pd.DataFrame:
    temp = pd.DataFrame({"actual": np.asarray(y_true), "prob": p})
    temp["bucket"] = pd.cut(
        temp["prob"],
        bins=[0, .40, .50, .60, .70, .80, .90, 1.0],
        include_lowest=True,
    )
    out = temp.groupby("bucket", observed=False).agg(
        predictions=("actual", "size"),
        avg_predicted_probability=("prob", "mean"),
        actual_success_rate=("actual", "mean"),
    )
    return out.reset_index()


def evaluate(split: SplitData, base, calibrator) -> pd.DataFrame:
    test = split.test.copy()
    test["prob"] = predict_calibrated(base, calibrator, test)
    test["pred"] = (test["prob"] >= 0.5).astype(int)

    auc = roc_auc_score(test["target"], test["prob"])
    brier = brier_score_loss(test["target"], test["prob"])
    acc = accuracy_score(test["target"], test["pred"])

    signals = test[test["prob"] >= SIGNAL_THRESHOLD].copy()
    signal_hit_rate = signals["target"].mean() if len(signals) else np.nan
    avg_signal_return = signals["future_return"].mean() if len(signals) else np.nan
    avg_net = avg_signal_return - ROUND_TRIP_COST if len(signals) else np.nan

    print("\n=== OUT-OF-SAMPLE TEST ===")
    print(f"Rows:                 {len(test):,}")
    print(f"ROC AUC:              {auc:.3f}")
    print(f"Brier score:          {brier:.4f}  (lower is better)")
    print(f"0.50 accuracy:        {acc:.1%}")
    print(f"Signals >= {SIGNAL_THRESHOLD:.0%}:     {len(signals):,}")
    if len(signals):
        print(f"Signal hit rate:      {signal_hit_rate:.1%}")
        print(f"Avg future return:    {avg_signal_return:.2%}")
        print(f"Avg after est. cost:  {avg_net:.2%}")

    cal = calibration_table(test["target"], test["prob"])
    print("\n=== CALIBRATION CHECK ===")
    print(cal.to_string(index=False))

    return test


def latest_scanner(raw: dict[str, pd.DataFrame], market: pd.DataFrame, base, calibrator):
    rows = []

    for ticker, px in raw.items():
        feat = add_features(px, market, ticker, include_target=False)
        feat = feat.dropna(subset=FEATURES)
        if feat.empty:
            continue

        latest = feat.iloc[[-1]].copy()
        prob = float(predict_calibrated(base, calibrator, latest)[0])

        rows.append({
            "ticker": ticker,
            "date": latest.index[0].date().isoformat(),
            "close": float(latest["Close"].iloc[0]),
            "prob_up_at_least_target": prob,
            "signal": "BUY-CANDIDATE" if prob >= SIGNAL_THRESHOLD else "NO SIGNAL",
            "rsi_14": float(latest["rsi_14"].iloc[0]),
            "vol_20": float(latest["vol_20"].iloc[0]),
        })

    result = pd.DataFrame(rows)
    if result.empty:
        return result

    result = result.sort_values("prob_up_at_least_target", ascending=False)
    return result


def main():
    print(
        f"Target: close >= {TARGET_RETURN:.1%} higher after "
        f"{HORIZON} trading days."
    )
    data, raw, market = build_dataset()
    split = chronological_split(data)

    print("\nSplit sizes:")
    print(f"  Train: {len(split.train):,}")
    print(f"  Calib: {len(split.calib):,}")
    print(f"  Test:  {len(split.test):,}")

    base, calibrator = fit_models(split)
    test = evaluate(split, base, calibrator)

    joblib.dump(base, MODEL_DIR / "base_model.joblib")
    joblib.dump(calibrator, MODEL_DIR / "calibrator.joblib")

    scan = latest_scanner(raw, market, base, calibrator)
    print("\n=== LATEST SCAN ===")
    if scan.empty:
        print("No scan results.")
    else:
        display = scan.copy()
        display["prob_up_at_least_target"] = (
            display["prob_up_at_least_target"] * 100
        ).map(lambda x: f"{x:.1f}%")
        display["close"] = display["close"].map(lambda x: f"{x:.2f}")
        display["rsi_14"] = display["rsi_14"].map(lambda x: f"{x:.1f}")
        display["vol_20"] = (display["vol_20"] * 100).map(lambda x: f"{x:.2f}%")
        print(display.to_string(index=False))

    test.to_csv("out_of_sample_predictions.csv")
    scan.to_csv("latest_scan.csv", index=False)

    print("\nSaved:")
    print("  models/base_model.joblib")
    print("  models/calibrator.joblib")
    print("  out_of_sample_predictions.csv")
    print("  latest_scan.csv")


if __name__ == "__main__":
    main()
