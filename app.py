
from __future__ import annotations

import os, math, json, sqlite3, warnings, datetime as dt
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="MT5 Day Trader AI V5", page_icon="⚡", layout="wide")

st.markdown("""
<style>
.block-container{padding-top:1rem;padding-bottom:4rem}
.card{padding:18px;border-radius:16px;border:1px solid rgba(120,120,120,.25);margin-bottom:12px}
.long{background:rgba(0,170,80,.09)} .short{background:rgba(180,70,70,.10)}
.wait{background:rgba(220,170,0,.08)} .big{font-size:1.75rem;font-weight:800}
.small{opacity:.75;font-size:.9rem}
</style>
""", unsafe_allow_html=True)

st.title("⚡ MT5 Day Trader AI V5")
st.caption(
    "Intraday research system for forex, crypto, indices and metals. "
    "Learns from multi-timeframe patterns, market regime, spread/volatility, "
    "walk-forward testing and stored paper-trade outcomes."
)

# -----------------------
# CONFIG
# -----------------------
TIMEFRAMES = {
    "M5": 5,
    "M15": 15,
    "M30": 30,
    "H1": 60,
}
HORIZON_BARS = [3, 6, 12, 24]
DEFAULT_SYMBOLS = "EURUSD, GBPUSD, USDJPY, XAUUSD, BTCUSD, ETHUSD"

with st.sidebar:
    st.header("Day-trading settings")
    symbols_text = st.text_area("MT5 symbols", DEFAULT_SYMBOLS, height=110)
    primary_tf = st.selectbox("Primary timeframe", list(TIMEFRAMES.keys()), index=1)
    target_rr = st.slider("Reward/Risk target", 1.0, 3.0, 1.5, 0.1)
    min_edge_pp = st.slider("Minimum probability edge (pp)", 2, 20, 6, 1)
    min_wf_auc = st.slider("Minimum walk-forward AUC", 0.50, 0.70, 0.56, 0.01)
    min_samples = st.slider("Minimum comparable samples", 20, 300, 60, 10)
    max_spread_atr = st.slider("Max spread as % of ATR", 1, 40, 15, 1) / 100
    data_mode = st.selectbox("Data source", ["MT5 bridge (recommended)", "Cloud fallback"])
    learn = st.button("Learn Intraday Market & Scan", type="primary", use_container_width=True)

    st.divider()
    st.caption("Live trading is disabled in this build.")
    st.caption("Use MT5 demo/paper validation first.")

DB_PATH = "v5_learning.db"

# -----------------------
# PERSISTENT LEARNING DB
# -----------------------
def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS predictions(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT,
            symbol TEXT,
            timeframe TEXT,
            horizon_bars INTEGER,
            side TEXT,
            probability REAL,
            baseline REAL,
            edge REAL,
            entry REAL,
            target REAL,
            stop REAL,
            resolved INTEGER DEFAULT 0,
            outcome INTEGER,
            realized_return REAL
        )
    """)
    conn.commit()
    return conn

def save_prediction(conn, row):
    conn.execute("""
        INSERT INTO predictions
        (ts,symbol,timeframe,horizon_bars,side,probability,baseline,edge,entry,target,stop)
        VALUES (?,?,?,?,?,?,?,?,?,?,?)
    """, (
        dt.datetime.utcnow().isoformat(),
        row["symbol"], row["timeframe"], int(row["horizon"]),
        row["decision"], float(row["prob"]), float(row["baseline"]),
        float(row["edge"]), float(row["close"]), float(row["target_price"]),
        float(row["stop_price"])
    ))
    conn.commit()

def learning_stats(conn, symbol):
    df = pd.read_sql_query(
        "SELECT * FROM predictions WHERE symbol=? AND resolved=1",
        conn, params=(symbol,)
    )
    if df.empty:
        return {"n":0,"hit":np.nan,"avg":np.nan}
    return {
        "n": len(df),
        "hit": float(df["outcome"].mean()) if "outcome" in df else np.nan,
        "avg": float(df["realized_return"].mean()) if "realized_return" in df else np.nan,
    }

# -----------------------
# MT5 BRIDGE
# -----------------------
def load_mt5_csv(symbol, timeframe):
    path = Path("mt5_data") / f"{symbol}_{timeframe}.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run mt5_bridge.py on the Windows PC running MT5."
        )
    df = pd.read_csv(path)
    if "time" not in df.columns:
        raise ValueError("MT5 CSV missing time column")
    df["time"] = pd.to_datetime(df["time"], utc=True).dt.tz_convert(None)
    df = df.set_index("time")
    rename = {
        "open":"Open","high":"High","low":"Low","close":"Close",
        "tick_volume":"Volume","spread":"Spread"
    }
    df = df.rename(columns=rename)
    for c in ["Open","High","Low","Close"]:
        if c not in df.columns:
            raise ValueError(f"Missing {c}")
    if "Volume" not in df.columns:
        df["Volume"] = 0.0
    if "Spread" not in df.columns:
        df["Spread"] = 0.0
    return df[["Open","High","Low","Close","Volume","Spread"]].copy()

# Cloud fallback lets the UI still run, but MT5 is the real target.
CLOUD_MAP = {
    "EURUSD":"EURUSD=X",
    "GBPUSD":"GBPUSD=X",
    "USDJPY":"JPY=X",
    "AUDUSD":"AUDUSD=X",
    "USDCAD":"CAD=X",
    "USDCHF":"CHF=X",
    "NZDUSD":"NZDUSD=X",
    "BTCUSD":"BTC-USD",
    "ETHUSD":"ETH-USD",
    "XAUUSD":"GC=F",
}

@st.cache_data(ttl=1800, show_spinner=False)
def cloud_download(symbol, timeframe):
    yf_symbol = CLOUD_MAP.get(symbol, symbol)
    interval = {"M5":"5m","M15":"15m","M30":"30m","H1":"60m"}[timeframe]
    period = "60d" if timeframe in ["M5","M15"] else "730d"
    df = yf.download(
        yf_symbol, period=period, interval=interval,
        auto_adjust=True, progress=False, threads=False
    )
    if df.empty:
        raise ValueError(f"No cloud data for {symbol}")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df["Spread"] = 0.0
    return df[["Open","High","Low","Close","Volume","Spread"]].copy()

def get_data(symbol, timeframe):
    if data_mode.startswith("MT5"):
        return load_mt5_csv(symbol, timeframe)
    return cloud_download(symbol, timeframe)

# -----------------------
# FEATURES
# -----------------------
def rsi(s,n=14):
    d=s.diff()
    up=d.clip(lower=0).ewm(alpha=1/n,adjust=False).mean()
    dn=(-d.clip(upper=0)).ewm(alpha=1/n,adjust=False).mean()
    rs=up/dn.replace(0,np.nan)
    return 100-100/(1+rs)

def atr(df,n=14):
    prev=df["Close"].shift(1)
    tr=pd.concat([
        df["High"]-df["Low"],
        (df["High"]-prev).abs(),
        (df["Low"]-prev).abs()
    ],axis=1).max(axis=1)
    return tr.rolling(n).mean()

def zscore(s,n):
    return (s-s.rolling(n).mean())/s.rolling(n).std().replace(0,np.nan)

def add_session_features(df):
    idx = df.index
    hour = idx.hour
    # UTC session approximations
    df["session_asia"] = ((hour >= 0) & (hour < 8)).astype(int)
    df["session_london"] = ((hour >= 7) & (hour < 16)).astype(int)
    df["session_ny"] = ((hour >= 12) & (hour < 21)).astype(int)
    df["session_overlap"] = ((hour >= 12) & (hour < 16)).astype(int)
    return df

FEATURES = [
    "r1","r2","r3","r6","r12","r24",
    "ema8","ema21","ema50","ema200",
    "trend_8_21","trend_21_50","trend_50_200",
    "vol6","vol12","vol24","atrp","rsi7","rsi14",
    "pz20","vz20","break20","break50","dd20",
    "spread_atr","body_pct","upper_wick","lower_wick",
    "session_asia","session_london","session_ny","session_overlap",
    "trend_score","momentum_score","breakout_score","meanrev_score"
]

def features(df, horizon, rr, include_target=True):
    x=df.copy()
    c=x["Close"]; r=c.pct_change()

    for n in [1,2,3,6,12,24]:
        x[f"r{n}"]=c.pct_change(n)

    for n in [8,21,50,200]:
        ema=c.ewm(span=n,adjust=False).mean()
        x[f"ema{n}"]=c/ema-1

    x["trend_8_21"]=c.ewm(span=8,adjust=False).mean()/c.ewm(span=21,adjust=False).mean()-1
    x["trend_21_50"]=c.ewm(span=21,adjust=False).mean()/c.ewm(span=50,adjust=False).mean()-1
    x["trend_50_200"]=c.ewm(span=50,adjust=False).mean()/c.ewm(span=200,adjust=False).mean()-1

    for n in [6,12,24]:
        x[f"vol{n}"]=r.rolling(n).std()*np.sqrt(252*24*60/TIMEFRAMES[primary_tf])

    x["ATR"]=atr(x,14)
    x["atrp"]=x["ATR"]/c
    x["rsi7"]=rsi(c,7); x["rsi14"]=rsi(c,14)
    x["pz20"]=zscore(c,20); x["vz20"]=zscore(x["Volume"].replace(0,np.nan),20)
    x["break20"]=c/c.rolling(20).max()
    x["break50"]=c/c.rolling(50).max()
    x["dd20"]=c/c.rolling(20).max()-1

    # Candlestick anatomy
    rng=(x["High"]-x["Low"]).replace(0,np.nan)
    x["body_pct"]=(x["Close"]-x["Open"]).abs()/rng
    x["upper_wick"]=(x["High"]-x[["Open","Close"]].max(axis=1))/rng
    x["lower_wick"]=(x[["Open","Close"]].min(axis=1)-x["Low"])/rng

    # MT5 spread is in broker points. Convert into a relative filter proxy.
    x["spread_atr"] = x["Spread"].abs() / (x["ATR"].replace(0,np.nan)*100000)

    x=add_session_features(x)

    x["trend_score"]=(x["trend_8_21"]>0).astype(int)+(x["trend_21_50"]>0).astype(int)+(x["trend_50_200"]>0).astype(int)
    x["momentum_score"]=(x["r6"]>0).astype(int)+(x["r12"]>0).astype(int)+(x["r24"]>0).astype(int)
    x["breakout_score"]=(x["break20"]>.995).astype(int)+(x["break50"]>.995).astype(int)
    x["meanrev_score"]=(x["rsi14"]<35).astype(int)+(x["pz20"]<-1).astype(int)

    if include_target:
        # Barrier-style target: LONG if TP is reached before SL within horizon.
        atrv=x["ATR"]
        target_dist=atrv*rr
        stop_dist=atrv

        future_high=pd.concat([x["High"].shift(-i) for i in range(1,horizon+1)],axis=1).max(axis=1)
        future_low=pd.concat([x["Low"].shift(-i) for i in range(1,horizon+1)],axis=1).min(axis=1)

        long_tp=c+target_dist
        long_sl=c-stop_dist
        short_tp=c-target_dist
        short_sl=c+stop_dist

        # Conservative barrier labeling: if both touched within window, mark ambiguous.
        long_hit=(future_high>=long_tp) & (future_low>long_sl)
        short_hit=(future_low<=short_tp) & (future_high<short_sl)

        x["target_long"]=long_hit.astype(float)
        x["target_short"]=short_hit.astype(float)
        x["future_return"]=c.shift(-horizon)/c-1
        x.loc[x["future_return"].isna(),["target_long","target_short"]]=np.nan

    return x

# -----------------------
# MODELS
# -----------------------
def make_lr():
    prep=ColumnTransformer([("num",Pipeline([
        ("imp",SimpleImputer(strategy="median")),
        ("sc",StandardScaler())
    ]),FEATURES)])
    return Pipeline([("prep",prep),("clf",LogisticRegression(
        C=.5,max_iter=2000,class_weight="balanced",random_state=42))])

def make_hgb():
    prep=ColumnTransformer([("num",SimpleImputer(strategy="median"),FEATURES)])
    return Pipeline([("prep",prep),("clf",HistGradientBoostingClassifier(
        learning_rate=.04,max_iter=220,max_leaf_nodes=15,
        min_samples_leaf=35,l2_regularization=2,random_state=42))])

def make_rf():
    prep=ColumnTransformer([("num",SimpleImputer(strategy="median"),FEATURES)])
    return Pipeline([("prep",prep),("clf",RandomForestClassifier(
        n_estimators=180,max_depth=7,min_samples_leaf=20,
        max_features=.55,class_weight="balanced",n_jobs=-1,random_state=42))])

def recency_weights(n):
    age=np.arange(n-1,-1,-1)
    return np.power(.5,age/1500)

def fit_ensemble(df, target):
    mods=[make_lr(),make_hgb(),make_rf()]
    w=recency_weights(len(df))
    for m in mods:
        try:m.fit(df[FEATURES],df[target],clf__sample_weight=w)
        except Exception:m.fit(df[FEATURES],df[target])
    return mods

def prob(mods,X):
    ps=[m.predict_proba(X[FEATURES])[:,1] for m in mods]
    return .30*ps[0]+.40*ps[1]+.30*ps[2]

def calibrate(p,y):
    p=np.clip(p,1e-6,1-1e-6)
    zz=np.log(p/(1-p)).reshape(-1,1)
    c=LogisticRegression(max_iter=1000)
    c.fit(zz,y)
    return c

def apply_cal(c,p):
    p=np.clip(p,1e-6,1-1e-6)
    zz=np.log(p/(1-p)).reshape(-1,1)
    return c.predict_proba(zz)[:,1]

def split(df):
    dates=np.array(sorted(df.index.unique()))
    a=dates[int(len(dates)*.60)]
    b=dates[int(len(dates)*.75)]
    return df[df.index<a],df[(df.index>=a)&(df.index<b)],df[df.index>=b]

def walk_forward(df,target,folds=4):
    dates=np.array(sorted(df.index.unique()))
    if len(dates)<1200:return np.nan
    bounds=np.linspace(.50,.90,folds+1)
    aucs=[]
    for i in range(folds):
        a=dates[int(len(dates)*bounds[i])]
        b=dates[int(len(dates)*bounds[i+1])]
        tr=df[df.index<a]
        te=df[(df.index>=a)&(df.index<b)]
        if len(tr)<700 or len(te)<150: continue
        try:
            mods=fit_ensemble(tr,target)
            p=prob(mods,te)
            if te[target].nunique()>1:
                aucs.append(roc_auc_score(te[target],p))
        except Exception:
            pass
    return float(np.mean(aucs)) if aucs else np.nan

def analyze_symbol(symbol, df, horizon):
    f=features(df,horizon,target_rr,True).dropna(subset=FEATURES+["target_long","target_short"])
    f["target_long"]=f["target_long"].astype(int)
    f["target_short"]=f["target_short"].astype(int)
    if len(f)<1500:
        return None

    tr,ca,te=split(f)
    if min(tr["target_long"].nunique(),ca["target_long"].nunique(),te["target_long"].nunique())<2:
        return None
    if min(tr["target_short"].nunique(),ca["target_short"].nunique(),te["target_short"].nunique())<2:
        return None

    long_mods=fit_ensemble(tr,"target_long")
    short_mods=fit_ensemble(tr,"target_short")
    long_cal=calibrate(prob(long_mods,ca),ca["target_long"])
    short_cal=calibrate(prob(short_mods,ca),ca["target_short"])

    lp_test=apply_cal(long_cal,prob(long_mods,te))
    sp_test=apply_cal(short_cal,prob(short_mods,te))

    long_auc=float(roc_auc_score(te["target_long"],lp_test))
    short_auc=float(roc_auc_score(te["target_short"],sp_test))
    long_wf=walk_forward(f,"target_long")
    short_wf=walk_forward(f,"target_short")

    base_long=float(pd.concat([tr,ca])["target_long"].mean())
    base_short=float(pd.concat([tr,ca])["target_short"].mean())

    latest=features(df,horizon,target_rr,False).dropna(subset=FEATURES).iloc[[-1]]
    lp=float(apply_cal(long_cal,prob(long_mods,latest))[0])
    sp=float(apply_cal(short_cal,prob(short_mods,latest))[0])

    long_edge=lp-base_long
    short_edge=sp-base_short

    # comparable test samples
    long_sel=(lp_test-base_long)>=max(long_edge-.01,0)
    short_sel=(sp_test-base_short)>=max(short_edge-.01,0)

    long_samples=int(long_sel.sum())
    short_samples=int(short_sel.sum())
    long_hit=float(te.loc[long_sel,"target_long"].mean()) if long_samples else np.nan
    short_hit=float(te.loc[short_sel,"target_short"].mean()) if short_samples else np.nan

    close=float(latest["Close"].iloc[0])
    atrv=float(latest["ATR"].iloc[0])
    spread_atr=float(latest["spread_atr"].iloc[0]) if not np.isnan(latest["spread_atr"].iloc[0]) else 0.0

    return {
        "symbol":symbol,"timeframe":primary_tf,"horizon":horizon,
        "long_prob":lp,"short_prob":sp,
        "long_baseline":base_long,"short_baseline":base_short,
        "long_edge":long_edge,"short_edge":short_edge,
        "long_auc":long_auc,"short_auc":short_auc,
        "long_wf":long_wf,"short_wf":short_wf,
        "long_samples":long_samples,"short_samples":short_samples,
        "long_hit":long_hit,"short_hit":short_hit,
        "close":close,"atr":atrv,"spread_atr":spread_atr,
        "rsi":float(latest["rsi14"].iloc[0]),
        "trend":float(latest["trend_score"].iloc[0]),
        "momentum":float(latest["momentum_score"].iloc[0]),
        "breakout":float(latest["breakout_score"].iloc[0]),
        "session_overlap":int(latest["session_overlap"].iloc[0]),
        "session_london":int(latest["session_london"].iloc[0]),
        "session_ny":int(latest["session_ny"].iloc[0]),
    }

def choose_decision(r):
    # Compare long vs short side, then enforce quality filters.
    if r["long_edge"] >= r["short_edge"]:
        side="LONG"
        p=r["long_prob"]; base=r["long_baseline"]; edge=r["long_edge"]
        auc=r["long_auc"]; wf=r["long_wf"]; samples=r["long_samples"]; hit=r["long_hit"]
        target=r["close"] + r["atr"]*target_rr
        stop=r["close"] - r["atr"]
    else:
        side="SHORT"
        p=r["short_prob"]; base=r["short_baseline"]; edge=r["short_edge"]
        auc=r["short_auc"]; wf=r["short_wf"]; samples=r["short_samples"]; hit=r["short_hit"]
        target=r["close"] - r["atr"]*target_rr
        stop=r["close"] + r["atr"]

    session_ok = bool(r["session_london"] or r["session_ny"] or r["session_overlap"])
    spread_ok = r["spread_atr"] <= max_spread_atr or r["spread_atr"] == 0.0
    quality = (
        edge >= min_edge_pp/100 and
        not np.isnan(wf) and wf >= min_wf_auc and
        auc >= .54 and
        samples >= min_samples and
        spread_ok and
        session_ok
    )

    if quality:
        decision=side
    elif edge > 0 and auc >= .52:
        decision="WAIT"
    else:
        decision="NO TRADE"

    return {
        **r,
        "decision":decision,
        "prob":p,"baseline":base,"edge":edge,
        "auc":auc,"wf_auc":wf,"samples":samples,"hit":hit,
        "target_price":target,"stop_price":stop
    }

# -----------------------
# RUN
# -----------------------
if learn:
    symbols=[s.strip().upper() for s in symbols_text.split(",") if s.strip()]
    conn=init_db()

    results=[]
    warnings_list=[]
    total=max(1,len(symbols)*len(HORIZON_BARS))
    done=0

    progress=st.progress(0,text="Reading intraday market data...")

    for symbol in symbols:
        try:
            df=get_data(symbol,primary_tf)
        except Exception as e:
            warnings_list.append(f"{symbol}: {e}")
            continue

        candidates=[]
        for h in HORIZON_BARS:
            done+=1
            progress.progress(min(.94,done/total*.94),text=f"Learning {symbol} — {h} bars...")
            try:
                a=analyze_symbol(symbol,df,h)
                if a: candidates.append(a)
            except Exception as e:
                warnings_list.append(f"{symbol} {h} bars: {e}")

        if candidates:
            # choose strongest side-adjusted walk-forward evidence
            def score(x):
                lw=-999 if np.isnan(x["long_wf"]) else x["long_wf"]
                sw=-999 if np.isnan(x["short_wf"]) else x["short_wf"]
                return max(lw,sw), max(x["long_edge"],x["short_edge"])
            best=sorted(candidates,key=score,reverse=True)[0]
            final=choose_decision(best)
            stats=learning_stats(conn,symbol)
            final["stored_n"]=stats["n"]
            final["stored_hit"]=stats["hit"]
            final["stored_avg"]=stats["avg"]
            results.append(final)

            if final["decision"] in ["LONG","SHORT"]:
                save_prediction(conn,final)

    progress.progress(1.0,text="Intraday learning cycle complete")

    if not results:
        st.error("No validated intraday models were produced.")
        if data_mode.startswith("MT5"):
            st.info("Run mt5_bridge.py on a Windows PC with MetaTrader 5 open, then upload/sync the mt5_data folder.")
        st.stop()

    res=pd.DataFrame(results)
    order={"LONG":0,"SHORT":0,"WAIT":1,"NO TRADE":2}
    res["_o"]=res["decision"].map(order)
    res=res.sort_values(["_o","edge","wf_auc"],ascending=[True,False,False]).drop(columns="_o")

    live_count=int(res["decision"].isin(["LONG","SHORT"]).sum())
    if live_count:
        st.success(f"{live_count} intraday setup(s) passed V5 filters.")
    else:
        st.warning("NO HIGH-CONFIDENCE DAY TRADE FOUND — V5 is refusing to force a setup.")

    st.subheader("MT5-style day-trade decisions")

    for _,r in res.iterrows():
        css="long" if r["decision"]=="LONG" else "short" if r["decision"]=="SHORT" else "wait"
        icon="🟢" if r["decision"]=="LONG" else "🔴" if r["decision"]=="SHORT" else "🟡"
        wf="N/A" if np.isnan(r["wf_auc"]) else f"{r['wf_auc']:.3f}"
        hit="N/A" if np.isnan(r["hit"]) else f"{r['hit']:.1%}"
        learned_hit="N/A" if np.isnan(r["stored_hit"]) else f"{r['stored_hit']:.1%}"

        st.markdown(f"""
        <div class="card {css}">
          <div style="font-size:1.25rem;font-weight:750">{icon} {r['symbol']} — {r['decision']}</div>
          <div class="big">{r['prob']:.1%} calibrated probability</div>
          <div>Baseline <b>{r['baseline']:.1%}</b> • Edge <b>{r['edge']*100:.1f} pp</b> • Horizon <b>{int(r['horizon'])} bars</b></div>
          <div>Walk-forward AUC <b>{wf}</b> • Test AUC <b>{r['auc']:.3f}</b> • Comparable samples <b>{int(r['samples'])}</b></div>
          <div>Historical hit <b>{hit}</b> • Stored paper outcomes <b>{int(r['stored_n'])}</b> • Stored hit <b>{learned_hit}</b></div>
          <div>Entry <b>{r['close']:.5f}</b> • TP <b>{r['target_price']:.5f}</b> • SL <b>{r['stop_price']:.5f}</b></div>
          <div class="small">RSI {r['rsi']:.1f} • Trend {r['trend']:.0f}/3 • Momentum {r['momentum']:.0f}/3 • Breakout {r['breakout']:.0f}/2 • Spread/ATR {r['spread_atr']:.2%}</div>
        </div>
        """,unsafe_allow_html=True)

    st.subheader("Learning engine")
    st.write(
        "V5 stores every qualifying paper-trade prediction in a local SQLite database. "
        "Once outcomes are resolved, those results can be used to re-weight symbols, timeframes, "
        "horizons and strategy families. This is the foundation for genuine persistent learning."
    )

    view=res.copy()
    for c in ["prob","baseline","edge","long_prob","short_prob","long_edge","short_edge"]:
        view[c]=view[c].map(lambda x:f"{x:.2%}")
    st.dataframe(
        view[["symbol","decision","timeframe","horizon","prob","baseline","edge","wf_auc","auc","samples","hit","close","target_price","stop_price"]],
        use_container_width=True,hide_index=True
    )

    st.download_button(
        "Download V5 scan CSV",
        res.to_csv(index=False).encode(),
        "mt5_day_trader_v5_scan.csv",
        "text/csv"
    )

    if warnings_list:
        with st.expander("Warnings"):
            for w in warnings_list: st.write(w)

    st.warning(
        "V5 is a research/paper-trading build. Live MT5 order execution is intentionally disabled. "
        "Validate on a demo account first, including broker spreads, slippage, commissions and news-event risk."
    )

else:
    st.markdown("""
### What V5 changes

V5 is built around **day trading**, not daily swing trading.

It supports:
- forex pairs such as `EURUSD`, `GBPUSD`, `USDJPY`;
- metals such as `XAUUSD`;
- crypto symbols such as `BTCUSD` and `ETHUSD`;
- intraday M5 / M15 / M30 / H1 data;
- LONG / SHORT / WAIT / NO TRADE;
- multi-horizon barrier testing;
- ATR-based stop and take-profit logic;
- trend, momentum, breakout and mean-reversion features;
- London / New York / overlap session awareness;
- spread-to-ATR filtering;
- recency-weighted machine learning;
- separate long and short models;
- walk-forward validation;
- persistent paper-trade history via SQLite.

### MT5 connection

For real MT5 data, run `mt5_bridge.py` on a **Windows PC with MetaTrader 5 installed and logged in**.  
The bridge exports the latest bars into the `mt5_data` folder, which this app reads.

The cloud Streamlit app itself cannot directly control an MT5 terminal running on your home PC.
""")
