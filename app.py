
from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

st.set_page_config(
    page_title="Market Probability AI V3",
    page_icon="🎯",
    layout="wide",
)

st.markdown("""
<style>
.block-container {padding-top:1.2rem;padding-bottom:4rem}
.card {
    padding:18px;border-radius:16px;border:1px solid rgba(120,120,120,.25);
    margin-bottom:12px;
}
.good {background:rgba(0,170,80,.09)}
.neutral {background:rgba(220,170,0,.08)}
.bad {background:rgba(220,60,60,.08)}
.big {font-size:1.85rem;font-weight:800}
.small {opacity:.75;font-size:.9rem}
</style>
""", unsafe_allow_html=True)

st.title("🎯 Market Probability AI V3")
st.caption(
    "Per-stock, multi-horizon edge finder. It compares model probability against each stock's own "
    "historical baseline and refuses to call a trade when the edge is weak."
)

DEFAULT_TICKERS = "AAPL, MSFT, NVDA, AMZN, META, GOOGL, JPM, COST"
HORIZONS = [1, 3, 5, 10, 20]

with st.sidebar:
    st.header("V3 settings")
    ticker_text = st.text_area("Tickers", DEFAULT_TICKERS, height=120)
    benchmark = st.text_input("Benchmark", "SPY")
    start_date = st.text_input("History start", "2012-01-01")
    target_mode = st.selectbox(
        "Target type",
        ["Volatility-adjusted", "Fixed %"],
        index=0,
    )
    fixed_target_pct = st.slider("Fixed target %", 0.25, 5.0, 1.0, 0.25)
    vol_target_mult = st.slider("Volatility target multiplier", 0.25, 1.50, 0.75, 0.05)
    min_edge_pp = st.slider("Minimum edge over baseline (percentage points)", 2, 20, 6, 1)
    min_wf_auc = st.slider("Minimum walk-forward AUC", 0.50, 0.70, 0.56, 0.01)
    min_samples = st.slider("Minimum historical signal samples", 20, 300, 60, 10)
    run = st.button("Find Real Edge", type="primary", use_container_width=True)

@st.cache_data(ttl=3600, show_spinner=False)
def download(ticker, start):
    df = yf.download(
        ticker, start=start, auto_adjust=True,
        progress=False, actions=False, threads=False
    )
    if df.empty:
        raise ValueError(f"No data for {ticker}")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    cols = ["Open","High","Low","Close","Volume"]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing {missing}")
    out = df[cols].copy()
    out.index = pd.to_datetime(out.index).tz_localize(None)
    return out

def rsi(s, n=14):
    d = s.diff()
    up = d.clip(lower=0).ewm(alpha=1/n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1/n, adjust=False).mean()
    rs = up / dn.replace(0, np.nan)
    return 100 - 100/(1+rs)

def atr(df, n=14):
    prev = df["Close"].shift(1)
    tr = pd.concat([
        df["High"]-df["Low"],
        (df["High"]-prev).abs(),
        (df["Low"]-prev).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(n).mean()

def zscore(s, n):
    m = s.rolling(n).mean()
    sd = s.rolling(n).std()
    return (s-m)/sd.replace(0,np.nan)

FEATURES = [
    "ret1","ret3","ret5","ret10","ret20","ret60",
    "ma10","ma20","ma50","ma200",
    "vol10","vol20","vol60","atrp",
    "rsi7","rsi14","pricez20","pricez60",
    "volz20","dd20","dd60","break20",
    "mkt1","mkt5","mkt20","mktvol20","mktma200",
    "rs5","rs20"
]

def features(px, market, horizon, target_mode, fixed_target, vol_mult, include_target=True):
    df = px.copy()
    c = df["Close"]
    r = c.pct_change()

    df["ret1"] = c.pct_change(1)
    df["ret3"] = c.pct_change(3)
    df["ret5"] = c.pct_change(5)
    df["ret10"] = c.pct_change(10)
    df["ret20"] = c.pct_change(20)
    df["ret60"] = c.pct_change(60)

    for n in [10,20,50,200]:
        df[f"ma{n}"] = c/c.rolling(n).mean()-1

    df["vol10"] = r.rolling(10).std()*np.sqrt(252)
    df["vol20"] = r.rolling(20).std()*np.sqrt(252)
    df["vol60"] = r.rolling(60).std()*np.sqrt(252)
    df["atrp"] = atr(df,14)/c
    df["rsi7"] = rsi(c,7)
    df["rsi14"] = rsi(c,14)
    df["pricez20"] = zscore(c,20)
    df["pricez60"] = zscore(c,60)
    df["volz20"] = zscore(df["Volume"],20)
    df["dd20"] = c/c.rolling(20).max()-1
    df["dd60"] = c/c.rolling(60).max()-1
    df["break20"] = c/c.rolling(20).max()

    m = market.copy()
    mc = m["Close"]
    mr = mc.pct_change()
    m["mkt1"] = mc.pct_change(1)
    m["mkt5"] = mc.pct_change(5)
    m["mkt20"] = mc.pct_change(20)
    m["mktvol20"] = mr.rolling(20).std()*np.sqrt(252)
    m["mktma200"] = mc/mc.rolling(200).mean()-1
    df = df.join(m[["mkt1","mkt5","mkt20","mktvol20","mktma200"]], how="left")

    df["rs5"] = df["ret5"] - df["mkt5"]
    df["rs20"] = df["ret20"] - df["mkt20"]

    if include_target:
        fwd = c.shift(-horizon)/c - 1
        if target_mode == "Fixed %":
            req = pd.Series(fixed_target, index=df.index)
        else:
            # Scale the required move to current volatility and horizon.
            # Uses only contemporaneous ATR, avoiding future information.
            req = (df["atrp"] * np.sqrt(max(horizon,1)) * vol_mult).clip(lower=0.0025)

        df["required_return"] = req
        df["future_return"] = fwd
        df["target"] = (fwd >= req).astype(float)
        df.loc[fwd.isna(), "target"] = np.nan

    return df

def make_model():
    prep = ColumnTransformer(
        [("num", Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("scale", StandardScaler())
        ]), FEATURES)],
        remainder="drop"
    )
    clf = LogisticRegression(
        C=0.6, max_iter=2000, class_weight="balanced", random_state=42
    )
    return Pipeline([("prep",prep),("clf",clf)])

def make_nonlinear():
    prep = ColumnTransformer(
        [("num", SimpleImputer(strategy="median"), FEATURES)],
        remainder="drop"
    )
    clf = HistGradientBoostingClassifier(
        learning_rate=0.04,
        max_iter=250,
        max_leaf_nodes=15,
        min_samples_leaf=35,
        l2_regularization=2.0,
        random_state=42
    )
    return Pipeline([("prep",prep),("clf",clf)])

def recency_weights(index):
    # Half-life ~2 years of trading days.
    n = len(index)
    if n == 0:
        return np.array([])
    age = np.arange(n-1, -1, -1)
    return np.power(0.5, age/504)

def ensemble_fit(train):
    m1 = make_model()
    m2 = make_nonlinear()
    # sklearn Pipeline supports fit params routed to final estimator.
    w = recency_weights(train.index)
    try:
        m1.fit(train[FEATURES], train["target"], clf__sample_weight=w)
    except Exception:
        m1.fit(train[FEATURES], train["target"])
    try:
        m2.fit(train[FEATURES], train["target"], clf__sample_weight=w)
    except Exception:
        m2.fit(train[FEATURES], train["target"])
    return m1, m2

def ensemble_prob(models, X):
    p1 = models[0].predict_proba(X[FEATURES])[:,1]
    p2 = models[1].predict_proba(X[FEATURES])[:,1]
    return 0.45*p1 + 0.55*p2

def calibrate(raw_p, y):
    raw_p = np.clip(raw_p,1e-6,1-1e-6)
    z = np.log(raw_p/(1-raw_p)).reshape(-1,1)
    cal = LogisticRegression(max_iter=1000)
    cal.fit(z,y)
    return cal

def apply_cal(cal, raw_p):
    raw_p = np.clip(raw_p,1e-6,1-1e-6)
    z = np.log(raw_p/(1-raw_p)).reshape(-1,1)
    return cal.predict_proba(z)[:,1]

def split_dates(df):
    dates = np.array(sorted(df.index.unique()))
    tr = dates[int(len(dates)*0.60)]
    ca = dates[int(len(dates)*0.75)]
    return (
        df[df.index < tr].copy(),
        df[(df.index >= tr)&(df.index < ca)].copy(),
        df[df.index >= ca].copy(),
    )

def walk_forward(df, folds=4):
    dates = np.array(sorted(df.index.unique()))
    if len(dates) < 700:
        return np.nan, 0, np.nan
    bounds = np.linspace(.50,.90,folds+1)
    aucs = []
    signal_hits = []
    signal_count = 0

    for i in range(folds):
        train_end = dates[int(len(dates)*bounds[i])]
        test_end = dates[int(len(dates)*bounds[i+1])]
        tr = df[df.index < train_end]
        te = df[(df.index >= train_end)&(df.index < test_end)]
        if len(tr) < 400 or len(te) < 80:
            continue
        try:
            models = ensemble_fit(tr)
            p = ensemble_prob(models, te)
            aucs.append(roc_auc_score(te["target"],p))
            baseline = tr["target"].mean()
            edge = p - baseline
            sel = edge >= 0.06
            if sel.any():
                signal_hits.extend(te.loc[sel,"target"].tolist())
                signal_count += int(sel.sum())
        except Exception:
            pass

    return (
        float(np.mean(aucs)) if aucs else np.nan,
        signal_count,
        float(np.mean(signal_hits)) if signal_hits else np.nan
    )

def evaluate_one(ticker, px, market, horizon, target_mode, fixed_target, vol_mult):
    df = features(px,market,horizon,target_mode,fixed_target,vol_mult,True)
    df = df.dropna(subset=FEATURES+["target","future_return","required_return"])
    df["target"] = df["target"].astype(int)
    if len(df) < 900:
        return None

    train, calib, test = split_dates(df)
    if min(train["target"].nunique(),calib["target"].nunique(),test["target"].nunique()) < 2:
        return None

    models = ensemble_fit(train)
    raw_cal = ensemble_prob(models, calib)
    cal = calibrate(raw_cal, calib["target"])
    p_test = apply_cal(cal, ensemble_prob(models,test))

    baseline = float(pd.concat([train,calib])["target"].mean())
    auc = float(roc_auc_score(test["target"],p_test))
    brier = float(brier_score_loss(test["target"],p_test))

    wf_auc, wf_signals, wf_hit = walk_forward(df, folds=4)

    latestf = features(px,market,horizon,target_mode,fixed_target,vol_mult,False)
    latestf = latestf.dropna(subset=FEATURES)
    if latestf.empty:
        return None
    latest = latestf.iloc[[-1]]
    current_p = float(apply_cal(cal, ensemble_prob(models,latest))[0])
    edge = current_p - baseline

    # Historical test signals at same or greater edge.
    test_edge = p_test - baseline
    comparable = test[test_edge >= max(edge-0.01, 0.0)].copy()
    comp_count = len(comparable)
    comp_hit = float(comparable["target"].mean()) if comp_count else np.nan
    comp_ret = float(comparable["future_return"].mean()) if comp_count else np.nan

    return {
        "ticker": ticker,
        "horizon": horizon,
        "prob": current_p,
        "baseline": baseline,
        "edge": edge,
        "auc": auc,
        "brier": brier,
        "wf_auc": wf_auc,
        "wf_signals": wf_signals,
        "wf_hit": wf_hit,
        "samples": comp_count,
        "historical_hit": comp_hit,
        "historical_return": comp_ret,
        "close": float(latest["Close"].iloc[0]),
        "rsi": float(latest["rsi14"].iloc[0]),
        "atrp": float(latest["atrp"].iloc[0]),
        "date": latest.index[0].date().isoformat(),
    }

def verdict(row):
    required_edge = min_edge_pp/100
    enough = row["samples"] >= min_samples
    wf_ok = not np.isnan(row["wf_auc"]) and row["wf_auc"] >= min_wf_auc
    edge_ok = row["edge"] >= required_edge
    auc_ok = row["auc"] >= 0.54

    if edge_ok and wf_ok and enough and auc_ok:
        return "LONG SETUP"
    if row["edge"] > 0 and row["auc"] >= 0.52:
        return "WATCH"
    return "NO TRADE"

if run:
    tickers = [x.strip().upper() for x in ticker_text.split(",") if x.strip()]
    if not tickers:
        st.error("Enter at least one ticker.")
        st.stop()

    fixed_target = fixed_target_pct/100
    progress = st.progress(0, text="Loading benchmark...")
    try:
        market = download(benchmark.upper(), start_date)
    except Exception as e:
        st.error(f"Benchmark failed: {e}")
        st.stop()

    results = []
    errors = []

    total_jobs = max(len(tickers)*len(HORIZONS),1)
    job = 0

    for ticker in tickers:
        try:
            px = download(ticker,start_date)
        except Exception as e:
            errors.append(f"{ticker}: {e}")
            continue

        per_ticker = []
        for horizon in HORIZONS:
            job += 1
            progress.progress(
                min(job/total_jobs*.95,.95),
                text=f"Testing {ticker} — {horizon} day horizon..."
            )
            try:
                r = evaluate_one(
                    ticker,px,market,horizon,
                    target_mode,fixed_target,vol_target_mult
                )
                if r:
                    per_ticker.append(r)
            except Exception as e:
                errors.append(f"{ticker} {horizon}d: {e}")

        if per_ticker:
            # Choose the best horizon using walk-forward AUC first, then current edge.
            best = sorted(
                per_ticker,
                key=lambda x: (
                    -999 if np.isnan(x["wf_auc"]) else x["wf_auc"],
                    x["auc"],
                    x["edge"]
                ),
                reverse=True
            )[0]
            best["verdict"] = verdict(best)
            results.append(best)

    progress.progress(1.0, text="Done")

    if not results:
        st.error("No models could be validated.")
        st.stop()

    res = pd.DataFrame(results)
    order = {"LONG SETUP":0,"WATCH":1,"NO TRADE":2}
    res["_o"] = res["verdict"].map(order)
    res = res.sort_values(
        ["_o","edge","wf_auc"],
        ascending=[True,False,False]
    ).drop(columns="_o")

    longs = int((res["verdict"]=="LONG SETUP").sum())
    st.subheader("Today's decision")
    if longs == 0:
        st.warning("NO TRADE TODAY — none of the tested stocks passed the V3 evidence thresholds.")
    else:
        st.success(f"{longs} stock(s) passed the V3 evidence thresholds.")

    st.subheader("Best validated setup per stock")

    for _,row in res.iterrows():
        cls = "good" if row["verdict"]=="LONG SETUP" else "neutral" if row["verdict"]=="WATCH" else "bad"
        icon = "🟢" if row["verdict"]=="LONG SETUP" else "🟡" if row["verdict"]=="WATCH" else "🔴"
        wf = "N/A" if np.isnan(row["wf_auc"]) else f"{row['wf_auc']:.3f}"
        hhit = "N/A" if np.isnan(row["historical_hit"]) else f"{row['historical_hit']:.1%}"
        hret = "N/A" if np.isnan(row["historical_return"]) else f"{row['historical_return']:.2%}"

        st.markdown(f"""
        <div class="card {cls}">
          <div style="font-size:1.25rem;font-weight:750">{icon} {row['ticker']} — {row['verdict']}</div>
          <div class="big">{row['prob']:.1%} model probability</div>
          <div>Historical baseline: <b>{row['baseline']:.1%}</b> &nbsp; | &nbsp;
               Edge: <b>{row['edge']*100:.1f} percentage points</b></div>
          <div>Best horizon: <b>{int(row['horizon'])} days</b> &nbsp; | &nbsp;
               Walk-forward AUC: <b>{wf}</b> &nbsp; | &nbsp;
               Test AUC: <b>{row['auc']:.3f}</b></div>
          <div>Comparable historical signals: <b>{int(row['samples'])}</b> &nbsp; | &nbsp;
               Hit rate: <b>{hhit}</b> &nbsp; | &nbsp;
               Avg realized return: <b>{hret}</b></div>
          <div class="small">Close {row['close']:.2f} • RSI {row['rsi']:.1f} • ATR {row['atrp']:.2%} • {row['date']}</div>
        </div>
        """, unsafe_allow_html=True)

    st.subheader("Validation table")
    table = res.copy()
    table["Probability"] = table["prob"].map(lambda x:f"{x:.1%}")
    table["Baseline"] = table["baseline"].map(lambda x:f"{x:.1%}")
    table["Edge"] = table["edge"].map(lambda x:f"{x*100:.1f} pp")
    table["WF AUC"] = table["wf_auc"].map(lambda x:"N/A" if np.isnan(x) else f"{x:.3f}")
    table["Test AUC"] = table["auc"].map(lambda x:f"{x:.3f}")
    table["Hit rate"] = table["historical_hit"].map(lambda x:"N/A" if np.isnan(x) else f"{x:.1%}")
    show = table[[
        "ticker","verdict","horizon","Probability","Baseline","Edge",
        "WF AUC","Test AUC","samples","Hit rate"
    ]].rename(columns={"ticker":"Ticker","verdict":"Decision","horizon":"Best horizon"})
    st.dataframe(show,use_container_width=True,hide_index=True)

    st.download_button(
        "Download V3 results CSV",
        data=res.to_csv(index=False).encode("utf-8"),
        file_name="market_probability_ai_v3_results.csv",
        mime="text/csv"
    )

    if errors:
        with st.expander("Warnings"):
            for e in errors:
                st.write(e)

    st.warning(
        "V3 is deliberately conservative. A LONG SETUP is still not a guarantee. "
        "Before real-money trading, add fees/slippage, portfolio sizing and paper trading."
    )

else:
    st.markdown("""
### What V3 changes

V3 no longer asks one model to predict everything.

For every stock it:
- trains a separate model;
- tests 1, 3, 5, 10 and 20-day horizons;
- chooses the horizon with the best walk-forward evidence;
- compares the current probability with that stock's historical baseline;
- weights recent market history more heavily;
- checks how similar historical high-edge signals actually performed;
- says **NO TRADE** when the evidence is weak.

The key number is now **edge over baseline**, not raw probability.

Example:

> Model probability: 61%  
> Stock's historical baseline: 48%  
> Edge: +13 percentage points

That is more meaningful than simply displaying “61%” by itself.
""")
