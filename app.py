
from __future__ import annotations
import time, json, os, warnings, datetime as dt
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
from streamlit_autorefresh import st_autorefresh

st.set_page_config(page_title="MT5 Day Trader AI V6 Live", page_icon="📡", layout="wide")

st.markdown("""
<style>
.block-container{padding-top:.7rem;padding-bottom:3rem}
.live-dot{display:inline-block;width:10px;height:10px;border-radius:50%;background:#18b663;margin-right:7px}
.price{font-size:2.35rem;font-weight:800}
.muted{opacity:.7}
.card{padding:14px;border:1px solid rgba(120,120,120,.25);border-radius:14px;margin-bottom:10px}
</style>
""", unsafe_allow_html=True)

st.title("📡 MT5 Day Trader AI V6 — Live Monitor")
st.caption("1-second dashboard refresh. True tick-by-tick prices require the MT5 bridge/broker feed.")

SYMBOLS_DEFAULT = "EURUSD, GBPUSD, USDJPY, XAUUSD, BTCUSD, ETHUSD"
TIMEFRAMES = ["M5","M15","M30","H1"]

with st.sidebar:
    st.header("Live monitor")
    symbols_text = st.text_area("Symbols", SYMBOLS_DEFAULT, height=105)
    selected = st.selectbox("Main chart symbol", [s.strip() for s in SYMBOLS_DEFAULT.split(",")])
    tf = st.selectbox("Chart timeframe", TIMEFRAMES, index=1)
    refresh_seconds = st.select_slider("Screen refresh", options=[1,2,5,10], value=1)
    source = st.selectbox("Live data source", ["MT5 live bridge", "Cloud fallback"])
    chart_bars = st.slider("Chart bars", 50, 500, 150, 25)
    auto_refresh = st.toggle("Live refresh", value=True)

if auto_refresh:
    st_autorefresh(interval=refresh_seconds*1000, key="live_market_refresh")

MT5_DIR = Path("mt5_data")
TICK_FILE = MT5_DIR / "live_ticks.json"

CLOUD_MAP = {
    "EURUSD":"EURUSD=X","GBPUSD":"GBPUSD=X","USDJPY":"JPY=X",
    "AUDUSD":"AUDUSD=X","USDCAD":"CAD=X","USDCHF":"CHF=X",
    "NZDUSD":"NZDUSD=X","BTCUSD":"BTC-USD","ETHUSD":"ETH-USD",
    "XAUUSD":"GC=F",
}

def load_live_ticks():
    if not TICK_FILE.exists():
        return {}
    try:
        return json.loads(TICK_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {}

def load_mt5_bars(symbol, timeframe):
    p=MT5_DIR/f"{symbol}_{timeframe}.csv"
    if not p.exists():
        raise FileNotFoundError(str(p))
    df=pd.read_csv(p)
    df["time"]=pd.to_datetime(df["time"],utc=True).dt.tz_convert(None)
    df=df.set_index("time")
    df=df.rename(columns={
        "open":"Open","high":"High","low":"Low","close":"Close",
        "tick_volume":"Volume","spread":"Spread"
    })
    return df

@st.cache_data(ttl=5, show_spinner=False)
def cloud_bars(symbol, timeframe):
    ys=CLOUD_MAP.get(symbol,symbol)
    interval={"M5":"5m","M15":"15m","M30":"30m","H1":"60m"}[timeframe]
    period="5d" if timeframe=="M5" else "60d"
    df=yf.download(ys,period=period,interval=interval,auto_adjust=True,progress=False,threads=False)
    if df.empty:
        raise ValueError("No cloud data")
    if isinstance(df.columns,pd.MultiIndex):
        df.columns=df.columns.get_level_values(0)
    df.index=pd.to_datetime(df.index).tz_localize(None)
    return df

def get_bars(symbol,timeframe):
    if source=="MT5 live bridge":
        return load_mt5_bars(symbol,timeframe)
    return cloud_bars(symbol,timeframe)

def current_quote(symbol):
    if source=="MT5 live bridge":
        ticks=load_live_ticks()
        t=ticks.get(symbol)
        if t:
            return {
                "bid":float(t.get("bid",0)),
                "ask":float(t.get("ask",0)),
                "last":float(t.get("last") or t.get("bid") or 0),
                "spread":float(t.get("ask",0))-float(t.get("bid",0)),
                "time":t.get("time",""),
                "source":"MT5",
            }
    try:
        df=get_bars(symbol,tf)
        last=float(df["Close"].iloc[-1])
        return {"bid":last,"ask":last,"last":last,"spread":0.0,
                "time":str(df.index[-1]),"source":"Cloud"}
    except Exception:
        return None

symbols=[s.strip().upper() for s in symbols_text.split(",") if s.strip()]
if selected not in symbols and symbols:
    selected=symbols[0]

# Header live status
ticks=load_live_ticks() if source=="MT5 live bridge" else {}
fresh=False
if source=="MT5 live bridge" and selected in ticks:
    try:
        tick_time=pd.to_datetime(ticks[selected].get("time"),utc=True)
        fresh=(pd.Timestamp.now(tz="UTC")-tick_time).total_seconds()<5
    except Exception:
        fresh=False

status_text="LIVE MT5" if fresh else ("MT5 WAITING" if source=="MT5 live bridge" else "CLOUD")
st.markdown(f'<div><span class="live-dot"></span><b>{status_text}</b> <span class="muted">• screen refresh {refresh_seconds}s • {dt.datetime.now().strftime("%H:%M:%S")}</span></div>', unsafe_allow_html=True)

# Quote strip
st.subheader("Live market board")
cols=st.columns(min(3,max(1,len(symbols))))
for i,sym in enumerate(symbols):
    q=current_quote(sym)
    with cols[i % len(cols)]:
        if q:
            st.markdown(f"""
            <div class="card">
              <b>{sym}</b>
              <div class="price">{q['last']:.5f}</div>
              <div>Bid {q['bid']:.5f} &nbsp; Ask {q['ask']:.5f}</div>
              <div class="muted">Spread {q['spread']:.5f} • {q['source']}</div>
            </div>
            """,unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='card'><b>{sym}</b><br>No live quote</div>",unsafe_allow_html=True)

# Main live chart
st.subheader(f"{selected} — {tf} live chart")
try:
    bars=get_bars(selected,tf).tail(chart_bars)
    fig=go.Figure(data=[go.Candlestick(
        x=bars.index,
        open=bars["Open"], high=bars["High"],
        low=bars["Low"], close=bars["Close"],
        name=selected
    )])
    q=current_quote(selected)
    if q:
        fig.add_hline(y=q["last"], line_dash="dot",
                      annotation_text=f"Live {q['last']:.5f}",
                      annotation_position="top left")
    fig.update_layout(
        height=520,
        margin=dict(l=5,r=5,t=20,b=5),
        xaxis_rangeslider_visible=False,
        showlegend=False,
    )
    st.plotly_chart(fig,use_container_width=True,config={"displayModeBar":False})
except Exception as e:
    st.warning(f"Chart data unavailable: {e}")

# Micro monitor
q=current_quote(selected)
if q:
    c1,c2,c3,c4=st.columns(4)
    c1.metric("Live/Bid",f"{q['bid']:.5f}")
    c2.metric("Ask",f"{q['ask']:.5f}")
    c3.metric("Spread",f"{q['spread']:.5f}")
    c4.metric("Feed",q["source"])

st.subheader("AI monitoring status")
st.info(
    "The live screen refreshes every second, but the AI model should not retrain every second. "
    "The correct architecture is: read ticks continuously → update live features → score the current "
    "market with the latest trained model → retrain on a slower schedule (for example every 15–60 minutes "
    "or after enough new bars arrive)."
)

if source=="MT5 live bridge" and not fresh:
    st.warning(
        "No fresh MT5 tick stream detected. Run the V6 mt5_bridge_live.py script on the Windows PC "
        "that has MetaTrader 5 open. It writes live bid/ask ticks every second."
    )

st.caption("Research/paper-trading system. Live order execution remains disabled.")
