
from __future__ import annotations
import warnings, math, datetime as dt
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Market Probability AI V4", page_icon="🤖", layout="wide")

st.markdown("""
<style>
.block-container{padding-top:1.1rem;padding-bottom:4rem}
.card{padding:18px;border-radius:16px;border:1px solid rgba(120,120,120,.25);margin-bottom:12px}
.buy{background:rgba(0,170,80,.09)} .watch{background:rgba(220,170,0,.08)}
.no{background:rgba(220,60,60,.08)} .big{font-size:1.8rem;font-weight:800}
.small{opacity:.75;font-size:.9rem}
</style>
""", unsafe_allow_html=True)

st.title("🤖 Market Probability AI V4")
st.caption(
    "Adaptive market-learning research system: price patterns + regimes + strategy ensemble + "
    "internet news sentiment + walk-forward validation. No AI can guarantee profit."
)

DEFAULT_TICKERS = "AAPL, MSFT, NVDA, AMZN, META, GOOGL, JPM, COST"
HORIZONS = [1, 3, 5, 10, 20]

with st.sidebar:
    st.header("Adaptive scanner")
    ticker_text = st.text_area("Tickers", DEFAULT_TICKERS, height=120)
    benchmark = st.text_input("Benchmark", "SPY")
    start_date = st.text_input("History start", "2012-01-01")
    min_edge_pp = st.slider("Minimum probability edge (pp)", 2, 20, 6, 1)
    min_auc = st.slider("Minimum walk-forward AUC", 0.50, 0.70, 0.56, 0.01)
    min_samples = st.slider("Minimum comparable samples", 20, 250, 60, 10)
    use_news = st.toggle("Use internet news sentiment", value=True)
    run = st.button("Learn Latest Market & Scan", type="primary", use_container_width=True)
    st.divider()
    st.caption("Retrains from the newest available market history each time the cached daily model expires.")
    st.caption("Examples: BHP.AX, CBA.AX, GC=F, BTC-USD")

@st.cache_data(ttl=21600, show_spinner=False)
def download(ticker, start):
    df = yf.download(ticker, start=start, auto_adjust=True, progress=False, actions=False, threads=False)
    if df.empty:
        raise ValueError(f"No data for {ticker}")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    cols = ["Open","High","Low","Close","Volume"]
    out = df[cols].copy()
    out.index = pd.to_datetime(out.index).tz_localize(None)
    return out

@st.cache_data(ttl=3600, show_spinner=False)
def news_sentiment(ticker):
    analyzer = SentimentIntensityAnalyzer()
    items = []
    try:
        news = yf.Ticker(ticker).news or []
    except Exception:
        news = []
    vals = []
    for n in news[:20]:
        title = ""
        if isinstance(n, dict):
            title = n.get("title") or n.get("content", {}).get("title") or ""
        if title:
            score = analyzer.polarity_scores(title)["compound"]
            vals.append(score)
            items.append((title, score))
    return (float(np.mean(vals)) if vals else 0.0), items

def rsi(s,n=14):
    d=s.diff()
    up=d.clip(lower=0).ewm(alpha=1/n,adjust=False).mean()
    dn=(-d.clip(upper=0)).ewm(alpha=1/n,adjust=False).mean()
    rs=up/dn.replace(0,np.nan)
    return 100-100/(1+rs)

def atr(df,n=14):
    p=df["Close"].shift(1)
    tr=pd.concat([(df["High"]-df["Low"]),
                  (df["High"]-p).abs(),
                  (df["Low"]-p).abs()],axis=1).max(axis=1)
    return tr.rolling(n).mean()

def z(s,n):
    return (s-s.rolling(n).mean())/s.rolling(n).std().replace(0,np.nan)

FEATURES=[
"r1","r3","r5","r10","r20","r60",
"ma10","ma20","ma50","ma200","vol10","vol20","vol60","atrp",
"rsi7","rsi14","pz20","pz60","vz20","dd20","dd60","break20","break60",
"m1","m5","m20","mv20","mma200","rs5","rs20",
"trend_score","momentum_score","meanrev_score","breakout_score","regime_score"
]

def feat(px,mkt,h,include_target=True):
    df=px.copy(); c=df["Close"]; r=c.pct_change()
    for n in [1,3,5,10,20,60]: df[f"r{n}"]=c.pct_change(n)
    for n in [10,20,50,200]: df[f"ma{n}"]=c/c.rolling(n).mean()-1
    for n in [10,20,60]: df[f"vol{n}"]=r.rolling(n).std()*np.sqrt(252)
    df["atrp"]=atr(df)/c
    df["rsi7"]=rsi(c,7); df["rsi14"]=rsi(c,14)
    df["pz20"]=z(c,20); df["pz60"]=z(c,60); df["vz20"]=z(df["Volume"],20)
    df["dd20"]=c/c.rolling(20).max()-1; df["dd60"]=c/c.rolling(60).max()-1
    df["break20"]=c/c.rolling(20).max(); df["break60"]=c/c.rolling(60).max()

    m=mkt.copy(); mc=m["Close"]; mr=mc.pct_change()
    m["m1"]=mc.pct_change(1); m["m5"]=mc.pct_change(5); m["m20"]=mc.pct_change(20)
    m["mv20"]=mr.rolling(20).std()*np.sqrt(252); m["mma200"]=mc/mc.rolling(200).mean()-1
    df=df.join(m[["m1","m5","m20","mv20","mma200"]],how="left")
    df["rs5"]=df["r5"]-df["m5"]; df["rs20"]=df["r20"]-df["m20"]

    # Strategy-family signals based only on information known at the time.
    df["trend_score"]=(df["ma20"]>0).astype(int)+(df["ma50"]>0).astype(int)+(df["ma200"]>0).astype(int)
    df["momentum_score"]=(df["r20"]>0).astype(int)+(df["r60"]>0).astype(int)+(df["rs20"]>0).astype(int)
    df["meanrev_score"]=((df["rsi14"]<35).astype(int)+(df["pz20"]<-1).astype(int))
    df["breakout_score"]=((df["break20"]>.995).astype(int)+(df["break60"]>.995).astype(int))
    df["regime_score"]=(df["mma200"]>0).astype(int)-(df["mv20"]>df["mv20"].rolling(252).quantile(.8)).astype(int)

    if include_target:
        fwd=c.shift(-h)/c-1
        req=(df["atrp"]*np.sqrt(max(h,1))*0.70).clip(lower=.0025)
        df["required"]=req; df["future_return"]=fwd
        df["target"]=(fwd>=req).astype(float)
        df.loc[fwd.isna(),"target"]=np.nan
    return df

def make_lr():
    prep=ColumnTransformer([("num",Pipeline([
        ("imp",SimpleImputer(strategy="median")),("sc",StandardScaler())
    ]),FEATURES)],remainder="drop")
    return Pipeline([("prep",prep),("clf",LogisticRegression(
        C=.6,max_iter=2000,class_weight="balanced",random_state=42))])

def make_hgb():
    prep=ColumnTransformer([("num",SimpleImputer(strategy="median"),FEATURES)],remainder="drop")
    return Pipeline([("prep",prep),("clf",HistGradientBoostingClassifier(
        learning_rate=.035,max_iter=260,max_leaf_nodes=15,
        min_samples_leaf=35,l2_regularization=2,random_state=42))])

def make_rf():
    prep=ColumnTransformer([("num",SimpleImputer(strategy="median"),FEATURES)],remainder="drop")
    return Pipeline([("prep",prep),("clf",RandomForestClassifier(
        n_estimators=220,max_depth=7,min_samples_leaf=20,max_features=.55,
        class_weight="balanced",n_jobs=-1,random_state=42))])

def weights(n):
    age=np.arange(n-1,-1,-1)
    return np.power(.5,age/504)

def fit_models(df):
    mods=[make_lr(),make_hgb(),make_rf()]
    w=weights(len(df))
    for m in mods:
        try:m.fit(df[FEATURES],df["target"],clf__sample_weight=w)
        except Exception:m.fit(df[FEATURES],df["target"])
    return mods

def raw_prob(mods,X):
    ps=[m.predict_proba(X[FEATURES])[:,1] for m in mods]
    return .30*ps[0]+.40*ps[1]+.30*ps[2]

def calibrator(p,y):
    p=np.clip(p,1e-6,1-1e-6); zz=np.log(p/(1-p)).reshape(-1,1)
    c=LogisticRegression(max_iter=1000); c.fit(zz,y); return c

def cal_prob(c,p):
    p=np.clip(p,1e-6,1-1e-6); zz=np.log(p/(1-p)).reshape(-1,1)
    return c.predict_proba(zz)[:,1]

def split(df):
    dates=np.array(sorted(df.index.unique()))
    a=dates[int(len(dates)*.60)]; b=dates[int(len(dates)*.75)]
    return df[df.index<a],df[(df.index>=a)&(df.index<b)],df[df.index>=b]

def walk_forward(df,folds=4):
    dates=np.array(sorted(df.index.unique()))
    if len(dates)<700:return np.nan
    bounds=np.linspace(.50,.90,folds+1); aucs=[]
    for i in range(folds):
        a=dates[int(len(dates)*bounds[i])]; b=dates[int(len(dates)*bounds[i+1])]
        tr=df[df.index<a]; te=df[(df.index>=a)&(df.index<b)]
        if len(tr)<400 or len(te)<80:continue
        try:
            mods=fit_models(tr); p=raw_prob(mods,te)
            aucs.append(roc_auc_score(te["target"],p))
        except Exception:pass
    return float(np.mean(aucs)) if aucs else np.nan

def analyze(ticker,px,mkt,h):
    df=feat(px,mkt,h,True).dropna(subset=FEATURES+["target","future_return","required"])
    df["target"]=df["target"].astype(int)
    if len(df)<900:return None
    tr,ca,te=split(df)
    if min(tr["target"].nunique(),ca["target"].nunique(),te["target"].nunique())<2:return None
    mods=fit_models(tr)
    cal=calibrator(raw_prob(mods,ca),ca["target"])
    pte=cal_prob(cal,raw_prob(mods,te))
    baseline=float(pd.concat([tr,ca])["target"].mean())
    auc=float(roc_auc_score(te["target"],pte))
    brier=float(brier_score_loss(te["target"],pte))
    wf=walk_forward(df)
    latest=feat(px,mkt,h,False).dropna(subset=FEATURES).iloc[[-1]]
    p=float(cal_prob(cal,raw_prob(mods,latest))[0])
    edge=p-baseline
    test_edge=pte-baseline
    similar=te[test_edge>=max(edge-.01,0)]
    return dict(
        horizon=h,prob=p,baseline=baseline,edge=edge,auc=auc,brier=brier,wf_auc=wf,
        samples=len(similar),
        hit=float(similar["target"].mean()) if len(similar) else np.nan,
        avgret=float(similar["future_return"].mean()) if len(similar) else np.nan,
        close=float(latest["Close"].iloc[0]),rsi=float(latest["rsi14"].iloc[0]),
        atrp=float(latest["atrp"].iloc[0]),
        trend=float(latest["trend_score"].iloc[0]),
        momentum=float(latest["momentum_score"].iloc[0]),
        meanrev=float(latest["meanrev_score"].iloc[0]),
        breakout=float(latest["breakout_score"].iloc[0]),
        regime=float(latest["regime_score"].iloc[0]),
        date=latest.index[0].date().isoformat()
    )

def decision(x,news_score):
    edge_ok=x["edge"]>=min_edge_pp/100
    wf_ok=not np.isnan(x["wf_auc"]) and x["wf_auc"]>=min_auc
    sample_ok=x["samples"]>=min_samples
    auc_ok=x["auc"]>=.54
    strategy_support=(x["trend"]+x["momentum"]+x["breakout"]+max(x["meanrev"],0))>=3
    news_ok=news_score>-0.45
    if edge_ok and wf_ok and sample_ok and auc_ok and strategy_support and news_ok:
        return "LONG SETUP"
    if x["edge"]>0 and x["auc"]>=.52:
        return "WATCH"
    return "NO TRADE"

if run:
    tickers=[x.strip().upper() for x in ticker_text.split(",") if x.strip()]
    if not tickers: st.error("Enter at least one ticker."); st.stop()

    prog=st.progress(0,text="Reading market data...")
    try:mkt=download(benchmark.upper(),start_date)
    except Exception as e: st.error(f"Benchmark failed: {e}"); st.stop()

    rows=[]; warnings_list=[]; total=max(1,len(tickers)*len(HORIZONS)); done=0
    for t in tickers:
        try:px=download(t,start_date)
        except Exception as e:
            warnings_list.append(f"{t}: {e}"); continue
        candidates=[]
        for h in HORIZONS:
            done+=1
            prog.progress(min(.92,done/total*.92),text=f"Learning {t}: {h}-day patterns...")
            try:
                a=analyze(t,px,mkt,h)
                if a:candidates.append(a)
            except Exception as e:
                warnings_list.append(f"{t} {h}d: {e}")

        if not candidates:continue
        best=sorted(candidates,key=lambda x:(
            -999 if np.isnan(x["wf_auc"]) else x["wf_auc"],x["auc"],x["edge"]
        ),reverse=True)[0]

        ns, headlines = news_sentiment(t) if use_news else (0.0,[])
        best["ticker"]=t; best["news_sentiment"]=ns
        best["headlines"]=headlines
        best["decision"]=decision(best,ns)
        rows.append(best)

    prog.progress(1.0,text="Learning cycle complete")
    if not rows: st.error("No validated models were produced."); st.stop()

    res=pd.DataFrame([{k:v for k,v in r.items() if k!="headlines"} for r in rows])
    order={"LONG SETUP":0,"WATCH":1,"NO TRADE":2}
    res["_o"]=res["decision"].map(order)
    res=res.sort_values(["_o","edge","wf_auc"],ascending=[True,False,False]).drop(columns="_o")

    st.caption(f"Latest learning cycle: {dt.datetime.now().strftime('%Y-%m-%d %H:%M')}")

    if (res["decision"]=="LONG SETUP").sum()==0:
        st.warning("NO HIGH-CONFIDENCE TRADE FOUND — V4 is refusing to force a signal.")
    else:
        st.success(f"{int((res['decision']=='LONG SETUP').sum())} setup(s) passed all evidence filters.")

    st.subheader("Adaptive AI decisions")
    lookup={r["ticker"]:r for r in rows}

    for _,r in res.iterrows():
        css="buy" if r["decision"]=="LONG SETUP" else "watch" if r["decision"]=="WATCH" else "no"
        icon="🟢" if r["decision"]=="LONG SETUP" else "🟡" if r["decision"]=="WATCH" else "🔴"
        wf="N/A" if np.isnan(r["wf_auc"]) else f"{r['wf_auc']:.3f}"
        hit="N/A" if np.isnan(r["hit"]) else f"{r['hit']:.1%}"
        avg="N/A" if np.isnan(r["avgret"]) else f"{r['avgret']:.2%}"
        sent="Positive" if r["news_sentiment"]>.15 else "Negative" if r["news_sentiment"]<-.15 else "Neutral"

        st.markdown(f"""
        <div class="card {css}">
        <div style="font-size:1.25rem;font-weight:750">{icon} {r['ticker']} — {r['decision']}</div>
        <div class="big">{r['prob']:.1%} calibrated probability</div>
        <div>Baseline <b>{r['baseline']:.1%}</b> • Edge <b>{r['edge']*100:.1f} pp</b> • Best horizon <b>{int(r['horizon'])}d</b></div>
        <div>Walk-forward AUC <b>{wf}</b> • Test AUC <b>{r['auc']:.3f}</b> • Samples <b>{int(r['samples'])}</b></div>
        <div>Historical hit <b>{hit}</b> • Avg return <b>{avg}</b> • News <b>{sent}</b> ({r['news_sentiment']:+.2f})</div>
        <div class="small">Trend {r['trend']:.0f}/3 • Momentum {r['momentum']:.0f}/3 • Breakout {r['breakout']:.0f}/2 • Mean-reversion {r['meanrev']:.0f}/2 • RSI {r['rsi']:.1f}</div>
        </div>
        """,unsafe_allow_html=True)

        info=lookup[r["ticker"]]
        if use_news and info["headlines"]:
            with st.expander(f"{r['ticker']} internet news intelligence"):
                for title,score in info["headlines"][:8]:
                    st.write(f"{score:+.2f} — {title}")

    st.subheader("Strategy intelligence")
    st.write(
        "V4 combines four strategy families rather than trusting a single pattern: "
        "**trend following, momentum/relative strength, breakout, and mean reversion**. "
        "The ML model decides whether those conditions historically improved the probability of the target "
        "for each stock and horizon."
    )

    st.subheader("Validation table")
    view=res.copy()
    view["Prob"]=view["prob"].map(lambda x:f"{x:.1%}")
    view["Baseline"]=view["baseline"].map(lambda x:f"{x:.1%}")
    view["Edge"]=view["edge"].map(lambda x:f"{x*100:.1f} pp")
    view["WF AUC"]=view["wf_auc"].map(lambda x:"N/A" if np.isnan(x) else f"{x:.3f}")
    view["Hit"]=view["hit"].map(lambda x:"N/A" if np.isnan(x) else f"{x:.1%}")
    st.dataframe(view[["ticker","decision","horizon","Prob","Baseline","Edge","WF AUC","auc","samples","Hit","news_sentiment"]],
                 use_container_width=True,hide_index=True)

    st.download_button("Download V4 results",res.to_csv(index=False).encode(),
                       "market_probability_ai_v4.csv","text/csv")

    if warnings_list:
        with st.expander("Warnings"):
            for w in warnings_list: st.write(w)

    st.warning(
        "This system re-learns from new historical data, but it is not autonomous guaranteed-profit AI. "
        "Real deployment still needs persistent storage, scheduled retraining, fees/slippage, paper trading, "
        "broker execution safeguards and monitoring for model decay."
    )
else:
    st.markdown("""
### What V4 adds

**1. Daily-adaptive learning**  
The app retrains against the newest available price history instead of relying on a permanently frozen model.

**2. Internet intelligence**  
It pulls current ticker news and calculates headline sentiment as an extra risk filter.

**3. Strategy ensemble**  
It evaluates trend following, momentum/relative strength, breakout and mean-reversion conditions together with machine learning.

**4. Per-stock, per-horizon learning**  
Each asset is tested separately across 1, 3, 5, 10 and 20 trading days.

**5. Evidence before trades**  
A LONG SETUP requires edge over baseline, walk-forward quality, enough comparable historical examples, model quality, strategy confirmation and no strongly negative news filter.

### Important
Streamlit Community Cloud does not guarantee permanent local storage. V4 therefore **re-trains from the complete latest dataset** rather than pretending it has permanent memory. A true 24/7 self-learning system needs a persistent database/model store plus a scheduled training service.
""")
