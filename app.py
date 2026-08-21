from __future__ import annotations

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestRegressor, VotingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss, accuracy_score, mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title='Market Probability AI V2', page_icon='🧠', layout='wide')
st.title('🧠 Market Probability AI V2')
st.caption('Calibrated probabilities, expected return, downside risk and walk-forward validation. No model can guarantee profit.')

st.markdown('''
<style>
.block-container {padding-top: 1.2rem; padding-bottom: 4rem;}
.card {padding:16px;border-radius:16px;margin:10px 0;border:1px solid rgba(128,128,128,.25)}
.buy {background:rgba(0,180,80,.10);border-color:rgba(0,180,80,.35)}
.hold {background:rgba(255,180,0,.10);border-color:rgba(255,180,0,.35)}
.avoid {background:rgba(220,60,60,.10);border-color:rgba(220,60,60,.35)}
.big {font-size:1.9rem;font-weight:800}
.small {opacity:.75;font-size:.9rem}
</style>
''', unsafe_allow_html=True)

DEFAULT_TICKERS = 'AAPL, MSFT, NVDA, AMZN, META, GOOGL, TSLA, JPM, XOM, COST'

with st.sidebar:
    st.header('Scanner settings')
    ticker_text = st.text_area('Tickers', DEFAULT_TICKERS, height=120)
    benchmark = st.text_input('Benchmark', 'SPY')
    start_date = st.text_input('History start', '2012-01-01')
    horizon = st.slider('Forecast horizon (trading days)', 1, 20, 5)
    target_pct = st.slider('Profit target', 0.25, 10.0, 1.0, 0.25)
    buy_prob_pct = st.slider('Minimum BUY probability', 50, 90, 65, 1)
    min_exp_pct = st.slider('Minimum expected return for BUY', -1.0, 5.0, 0.5, 0.25)
    max_down_pct = st.slider('Max downside probability for BUY', 10, 70, 45, 1)
    run = st.button('Train V2 & Scan', type='primary', use_container_width=True)
    st.divider()
    st.caption('ASX examples: BHP.AX, CBA.AX, FMG.AX')
    st.caption('Gold: GLD or GC=F')
    st.caption('Crypto: BTC-USD')

target_return = target_pct / 100
buy_threshold = buy_prob_pct / 100
min_expected = min_exp_pct / 100
max_down = max_down_pct / 100

@st.cache_data(ttl=3600, show_spinner=False)
def download(ticker: str, start: str) -> pd.DataFrame:
    df = yf.download(ticker, start=start, auto_adjust=True, progress=False, actions=False, threads=False)
    if df.empty:
        raise ValueError(f'No data returned for {ticker}')
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    cols = ['Open','High','Low','Close','Volume']
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f'{ticker} missing {missing}')
    df = df[cols].copy()
    df.index = pd.to_datetime(df.index).tz_localize(None)
    return df

def rsi(close: pd.Series, n=14):
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1/n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1/n, adjust=False).mean()
    rs = up / dn.replace(0, np.nan)
    return 100 - 100/(1+rs)

def atr(df: pd.DataFrame, n=14):
    prev = df['Close'].shift(1)
    tr = pd.concat([(df['High']-df['Low']), (df['High']-prev).abs(), (df['Low']-prev).abs()], axis=1).max(axis=1)
    return tr.rolling(n).mean()

def zscore(s, n):
    return (s-s.rolling(n).mean()) / s.rolling(n).std().replace(0, np.nan)

def regime_series(market):
    c = market['Close']
    ma50 = c.rolling(50).mean(); ma200 = c.rolling(200).mean()
    r20 = c.pct_change(20)
    vol = c.pct_change().rolling(20).std()*np.sqrt(252)
    q80 = vol.rolling(252).quantile(.80)
    out = pd.Series('Neutral', index=market.index, dtype='object')
    out[(c>ma200)&(ma50>ma200)&(r20>0)] = 'Bull'
    out[(c<ma200)&(ma50<ma200)&(r20<0)] = 'Bear'
    out[vol>q80] = 'High Volatility'
    return out

def features(price, market, ticker, include_target=True):
    df = price.copy(); c = df['Close']; r1 = c.pct_change()
    for n in [1,2,3,5,10,20,60,120]: df[f'ret_{n}'] = c.pct_change(n)
    for n in [5,10,20,50,100,200]: df[f'ma_{n}_dist'] = c/c.rolling(n).mean()-1
    df['trend_5_20'] = c.rolling(5).mean()/c.rolling(20).mean()-1
    df['trend_20_50'] = c.rolling(20).mean()/c.rolling(50).mean()-1
    df['trend_50_200'] = c.rolling(50).mean()/c.rolling(200).mean()-1
    for n in [5,10,20,60]: df[f'vol_{n}'] = r1.rolling(n).std()*np.sqrt(252)
    df['range_1'] = (df['High']-df['Low'])/c
    df['gap_1'] = df['Open']/c.shift(1)-1
    df['atr_14_pct'] = atr(df)/c
    df['rsi_14'] = rsi(c,14); df['rsi_7'] = rsi(c,7)
    df['drawdown_20'] = c/c.rolling(20).max()-1
    df['drawdown_60'] = c/c.rolling(60).max()-1
    df['breakout_20'] = c/c.rolling(20).max()
    df['breakout_60'] = c/c.rolling(60).max()
    df['volume_z_20'] = zscore(df['Volume'],20)
    df['volume_z_60'] = zscore(df['Volume'],60)
    df['price_z_20'] = zscore(c,20); df['price_z_60'] = zscore(c,60)

    m = market.copy(); mc = m['Close']; mr = mc.pct_change()
    m['market_ret_1'] = mc.pct_change(1); m['market_ret_5'] = mc.pct_change(5); m['market_ret_20'] = mc.pct_change(20)
    m['market_vol_20'] = mr.rolling(20).std()*np.sqrt(252)
    m['market_ma200_dist'] = mc/mc.rolling(200).mean()-1
    m['market_rsi_14'] = rsi(mc,14)
    df = df.join(m[['market_ret_1','market_ret_5','market_ret_20','market_vol_20','market_ma200_dist','market_rsi_14']], how='left')
    df['rel_strength_5'] = df['ret_5']-df['market_ret_5']
    df['rel_strength_20'] = df['ret_20']-df['market_ret_20']
    reg = regime_series(market).reindex(df.index).fillna('Neutral')
    df['regime'] = reg
    df['regime_bull'] = (reg=='Bull').astype(int)
    df['regime_bear'] = (reg=='Bear').astype(int)
    df['regime_highvol'] = (reg=='High Volatility').astype(int)
    df['ticker'] = ticker
    if include_target:
        fr = c.shift(-horizon)/c-1
        df['future_return'] = fr
        df['target_up'] = (fr>=target_return).astype(float)
        df['target_down'] = (fr<=-target_return).astype(float)
        df.loc[fr.isna(), ['target_up','target_down']] = np.nan
    return df

FEATURES = [
'ret_1','ret_2','ret_3','ret_5','ret_10','ret_20','ret_60','ret_120',
'ma_5_dist','ma_10_dist','ma_20_dist','ma_50_dist','ma_100_dist','ma_200_dist',
'trend_5_20','trend_20_50','trend_50_200','vol_5','vol_10','vol_20','vol_60',
'range_1','gap_1','atr_14_pct','rsi_14','rsi_7','drawdown_20','drawdown_60',
'breakout_20','breakout_60','volume_z_20','volume_z_60','price_z_20','price_z_60',
'market_ret_1','market_ret_5','market_ret_20','market_vol_20','market_ma200_dist','market_rsi_14',
'rel_strength_5','rel_strength_20','regime_bull','regime_bear','regime_highvol']

def classifier():
    prep = ColumnTransformer([('num', Pipeline([('imp',SimpleImputer(strategy='median')),('sc',StandardScaler())]), FEATURES)], remainder='drop')
    lr = LogisticRegression(C=.5, max_iter=2000, class_weight='balanced', random_state=42)
    hgb = HistGradientBoostingClassifier(learning_rate=.035, max_iter=350, max_leaf_nodes=15, min_samples_leaf=50, l2_regularization=2.0, random_state=42)
    vote = VotingClassifier([('lr',lr),('hgb',hgb)], voting='soft', weights=[1,2])
    return Pipeline([('prep',prep),('model',vote)])

def regressor():
    prep = ColumnTransformer([('num',SimpleImputer(strategy='median'),FEATURES)], remainder='drop')
    rf = RandomForestRegressor(n_estimators=300,max_depth=8,min_samples_leaf=20,max_features=.6,n_jobs=-1,random_state=42)
    return Pipeline([('prep',prep),('model',rf)])

def split(data):
    dates = np.array(sorted(data.index.unique()))
    d1 = dates[int(len(dates)*.65)]; d2 = dates[int(len(dates)*.80)]
    return data[data.index<d1], data[(data.index>=d1)&(data.index<d2)], data[data.index>=d2]

def fit_cal(base, X, y):
    p = np.clip(base.predict_proba(X)[:,1],1e-6,1-1e-6)
    z = np.log(p/(1-p)).reshape(-1,1)
    cal = LogisticRegression(max_iter=1000,random_state=42).fit(z,y)
    return cal

def cal_prob(base, cal, X):
    p = np.clip(base.predict_proba(X)[:,1],1e-6,1-1e-6)
    z = np.log(p/(1-p)).reshape(-1,1)
    return cal.predict_proba(z)[:,1]

def walk_forward(data, folds=4):
    dates = np.array(sorted(data.index.unique()))
    if len(dates)<800: return np.nan, []
    cuts = np.linspace(.55,.90,folds+1); scores=[]
    for i in range(folds):
        a=dates[int(len(dates)*cuts[i])]; b=dates[int(len(dates)*cuts[i+1])]
        tr=data[data.index<a]; te=data[(data.index>=a)&(data.index<b)]
        if len(tr)<1000 or len(te)<100: continue
        try:
            m=classifier().fit(tr[FEATURES],tr['target_up'])
            scores.append(roc_auc_score(te['target_up'],m.predict_proba(te[FEATURES])[:,1]))
        except Exception: pass
    return (float(np.mean(scores)) if scores else np.nan), scores

def signal(prob, down, exp):
    if prob>=buy_threshold and down<=max_down and exp>=min_expected: return 'BUY'
    if prob>=.50 and exp>-0.002: return 'HOLD'
    return 'AVOID'

def confidence(prob, down, exp, auc):
    s=max(0,prob-.5)*3 + max(0,prob-down)*1.5 + max(0,exp)*8 + max(0,auc-.5)*2.5
    return 'High' if s>=.75 else 'Moderate' if s>=.40 else 'Low'

st.info(f'Target: at least **{target_pct:.2f}% higher in {horizon} trading days**. BUY also requires expected return ≥ **{min_exp_pct:.2f}%** and downside probability ≤ **{max_down_pct}%**.')

if run:
    tickers=[x.strip().upper() for x in ticker_text.split(',') if x.strip()]
    if not tickers: st.error('Enter at least one ticker.'); st.stop()
    bar=st.progress(0,text='Downloading benchmark...')
    try: market=download(benchmark.upper(),start_date)
    except Exception as e: st.error(f'Benchmark failed: {e}'); st.stop()
    frames=[]; raw={}; errors=[]
    for i,t in enumerate(tickers,1):
        bar.progress(min(.45,i/max(len(tickers),1)*.45),text=f'Loading {t}...')
        try:
            px=download(t,start_date); raw[t]=px; frames.append(features(px,market,t,True))
        except Exception as e: errors.append(f'{t}: {e}')
    if not frames: st.error('No usable ticker data was downloaded.'); st.stop()
    data=pd.concat(frames).sort_index().dropna(subset=FEATURES+['target_up','target_down','future_return'])
    data['target_up']=data['target_up'].astype(int); data['target_down']=data['target_down'].astype(int)
    if len(data.index.unique())<700: st.error('Not enough history. Use an earlier start date.'); st.stop()
    train,calib,test=split(data)

    bar.progress(.55,text='Training upside ensemble...')
    up=classifier().fit(train[FEATURES],train['target_up']); upcal=fit_cal(up,calib[FEATURES],calib['target_up'])
    bar.progress(.65,text='Training downside ensemble...')
    dn=classifier().fit(train[FEATURES],train['target_down']); dncal=fit_cal(dn,calib[FEATURES],calib['target_down'])
    bar.progress(.75,text='Training expected-return model...')
    reg=regressor().fit(pd.concat([train,calib])[FEATURES],pd.concat([train,calib])['future_return'])

    test=test.copy(); test['up_prob']=cal_prob(up,upcal,test[FEATURES]); test['down_prob']=cal_prob(dn,dncal,test[FEATURES]); test['expected_return']=reg.predict(test[FEATURES])
    auc=roc_auc_score(test['target_up'],test['up_prob']); brier=brier_score_loss(test['target_up'],test['up_prob']); acc=accuracy_score(test['target_up'],(test['up_prob']>=.5).astype(int)); mae=mean_absolute_error(test['future_return'],test['expected_return'])
    wf_auc,wf=walk_forward(data)
    test['signal']=[signal(p,d,e) for p,d,e in zip(test['up_prob'],test['down_prob'],test['expected_return'])]
    buys=test[test['signal']=='BUY']; buy_hit=buys['target_up'].mean() if len(buys) else np.nan; buy_ret=buys['future_return'].mean() if len(buys) else np.nan

    bar.progress(.90,text='Scanning latest market state...')
    rows=[]
    for t,px in raw.items():
        f=features(px,market,t,False).dropna(subset=FEATURES)
        if f.empty: continue
        x=f.iloc[[-1]]; p=float(cal_prob(up,upcal,x[FEATURES])[0]); d=float(cal_prob(dn,dncal,x[FEATURES])[0]); e=float(reg.predict(x[FEATURES])[0])
        sig=signal(p,d,e); conf=confidence(p,d,e,auc); atrp=float(x['atr_14_pct'].iloc[0]); stop=-max(atrp*1.5,.01); take=max(target_return,abs(stop)*1.5)
        rows.append({'Ticker':t,'Signal':sig,'Profit probability':p,'Downside probability':d,'Expected return':e,'Confidence':conf,'Regime':str(x['regime'].iloc[0]),'RSI':float(x['rsi_14'].iloc[0]),'Close':float(x['Close'].iloc[0]),'Research stop %':stop,'Research take-profit %':take,'Model edge':p-d})
    scan=pd.DataFrame(rows)
    if scan.empty: st.error('No latest scan results.'); st.stop()
    order={'BUY':0,'HOLD':1,'AVOID':2}; scan['_o']=scan['Signal'].map(order); scan=scan.sort_values(['_o','Profit probability','Expected return'],ascending=[True,False,False]).drop(columns='_o')
    bar.progress(1.0,text='Done')

    st.subheader('Model validation')
    a,b,c,d=st.columns(4); a.metric('Out-of-sample ROC AUC',f'{auc:.3f}'); b.metric('Walk-forward ROC AUC','N/A' if np.isnan(wf_auc) else f'{wf_auc:.3f}'); c.metric('Brier score',f'{brier:.3f}'); d.metric('Return MAE',f'{mae:.2%}')
    a,b,c=st.columns(3); a.metric('Accuracy',f'{acc:.1%}'); b.metric('Historical BUY signals',f'{len(buys):,}'); c.metric('BUY hit rate','N/A' if np.isnan(buy_hit) else f'{buy_hit:.1%}')
    if not np.isnan(buy_ret): st.caption(f'Average realized {horizon}-day return on historical BUY signals: {buy_ret:.2%}')
    if auc<.55: st.warning('Current predictive power is weak. Treat signals as research only.')
    elif auc<.62: st.info('Current predictive power is modest. Keep validating before trusting live signals.')
    else: st.success('Stronger out-of-sample discrimination detected. Continue paper-trading before real-money use.')

    st.subheader('Latest AI scanner')
    for _,r in scan.iterrows():
        cls='buy' if r['Signal']=='BUY' else 'hold' if r['Signal']=='HOLD' else 'avoid'; icon='🟢' if r['Signal']=='BUY' else '🟡' if r['Signal']=='HOLD' else '🔴'
        st.markdown(f'''<div class="card {cls}"><b>{icon} {r['Ticker']} — {r['Signal']}</b><div class="big">{r['Profit probability']:.1%} profit probability</div><div>Expected return: <b>{r['Expected return']:.2%}</b> &nbsp; | &nbsp; Downside probability: <b>{r['Downside probability']:.1%}</b></div><div>Confidence: <b>{r['Confidence']}</b> &nbsp; | &nbsp; Regime: <b>{r['Regime']}</b> &nbsp; | &nbsp; RSI: <b>{r['RSI']:.1f}</b></div><div class="small">Model edge: {r['Model edge']:.1%} • Research stop: {r['Research stop %']:.2%} • Research take-profit: {r['Research take-profit %']:.2%}</div></div>''',unsafe_allow_html=True)

    st.subheader('Full scanner table')
    table=scan.copy()
    for col in ['Profit probability','Downside probability','Expected return','Research stop %','Research take-profit %','Model edge']: table[col]=table[col].map(lambda x:f'{x:.2%}')
    table['Close']=table['Close'].map(lambda x:f'{x:,.2f}'); table['RSI']=table['RSI'].map(lambda x:f'{x:.1f}')
    st.dataframe(table,use_container_width=True,hide_index=True)
    st.download_button('Download V2 scan CSV',scan.to_csv(index=False).encode(),'market_probability_ai_v2_scan.csv','text/csv')

    st.subheader('Probability calibration')
    cal=test.copy(); cal['bucket']=pd.cut(cal['up_prob'],bins=np.linspace(0,1,11),include_lowest=True)
    out=cal.groupby('bucket',observed=False).agg(Predictions=('target_up','size'),Avg_predicted_probability=('up_prob','mean'),Actual_success_rate=('target_up','mean'),Avg_realized_return=('future_return','mean')).reset_index()
    st.dataframe(out,use_container_width=True,hide_index=True)
    with st.expander('Walk-forward fold scores'):
        st.write(pd.DataFrame({'Fold':range(1,len(wf)+1),'ROC AUC':wf}) if wf else 'Not enough history for fold scores.')
    if errors:
        with st.expander('Ticker warnings'):
            for e in errors: st.write(e)
    st.warning('BUY is not a guarantee. Real trading also requires execution costs, slippage, position sizing, portfolio limits, tax considerations and live paper-trading validation.')
else:
    st.markdown('''
### What V2 improves
This version combines multiple model types and refuses to call something a BUY unless **probability, expected return and downside risk** agree.

It adds stronger momentum, trend, volatility, ATR, drawdown, breakout, relative-strength, market-regime and volume features, plus **walk-forward validation** so the app is harder to fool with one lucky historical period.

A good first test is `AAPL, MSFT, NVDA, AMZN, META, GOOGL, JPM, COST` with a 5-day horizon and 1% target.
''')
