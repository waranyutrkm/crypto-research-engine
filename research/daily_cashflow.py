import pandas as pd, numpy as np, requests, time, json
from pathlib import Path
from itertools import combinations
import os
EXCHANGE=os.getenv("RESEARCH_EXCHANGE","binance")
OUT=Path("results"); OUT.mkdir(exist_ok=True)
SYMS=["BTCUSDT","ETHUSDT","BNBUSDT","SOLUSDT","XRPUSDT"]
BASE="https://fapi.binance.com"
def get(path,params):
    for i in range(4):
        try:
            r=requests.get(BASE+path,params=params,timeout=30); r.raise_for_status(); return r.json()
        except requests.RequestException:
            if i==3: raise
            time.sleep(2**i)
from io import BytesIO
from zipfile import ZipFile
from concurrent.futures import ThreadPoolExecutor
def archive(sym,kind,month,interval="1h"):
    base="https://data.binance.vision/data/futures/um/monthly"
    suffix=f"{sym}-{interval}-{month}" if kind=="klines" else f"{sym}-fundingRate-{month}"
    folder=f"{sym}/{interval}" if kind=="klines" else sym
    url=f"{base}/{kind}/{folder}/{suffix}.zip"
    for i in range(3):
        try:
            response=requests.get(url,timeout=60)
            response.raise_for_status()
            with ZipFile(BytesIO(response.content)) as z:
                return pd.read_csv(z.open(z.namelist()[0]),header=None)
        except (requests.RequestException, OSError) as exc:
            if i==2: raise RuntimeError(url) from exc
            time.sleep(2**i)
def monthly(sym,kind,interval="1h"):
    end=pd.Timestamp.now(tz="UTC").tz_localize(None).to_period("M")-1
    months=[str(m) for m in pd.period_range("2021-01",end,freq="M")]
    with ThreadPoolExecutor(max_workers=6) as pool:
        frames=list(pool.map(lambda m:archive(sym,kind,m,interval),months))
    return pd.concat(frames,ignore_index=True)
def load(sym,interval="1h"):
    d=monthly(sym,"klines",interval)
    ts=pd.to_numeric(d[0],errors="coerce")
    d=d.loc[ts.notna()].copy()
    d.index=pd.to_datetime(ts.dropna().astype("int64"),unit="ms",utc=True)
    return pd.to_numeric(d[4],errors="raise").rename(sym)
def funding(sym):
    d=monthly(sym,"fundingRate")
    ts=pd.to_numeric(d[0],errors="coerce")
    d=d.loc[ts.notna()].copy()
    d.index=pd.to_datetime(ts.dropna().astype("int64"),unit="ms",utc=True)
    return pd.to_numeric(d[2],errors="raise").resample("1h").sum().rename(sym)
def metrics(r):
    d=(1+r).resample("1D").prod()-1; e=(1+d).cumprod(); dd=e/e.cummax().clip(lower=1)-1
    return dict(days=len(d),cagr=float(e.iloc[-1]**(365/len(d))-1),sharpe=float(d.mean()/d.std()*np.sqrt(365)) if d.std()>0 else None,max_dd=float(dd.min()),positive_days=float((d>0).mean()),median_daily=float(d.median()),worst_day=float(d.min()),profit_factor=float(d.clip(lower=0).sum()/(-d.clip(upper=0).sum())) if (d<0).any() else None)
if EXCHANGE=="binance":
    p=pd.concat([load(s) for s in SYMS],axis=1).sort_index().asfreq("1h")
    assert not p.isna().any().any(),"missing prices"
    f=pd.concat([funding(s) for s in SYMS],axis=1).reindex(p.index).fillna(0)
else:
    from exchange_data import fetch_exchange
    p,f,coverage=fetch_exchange(EXCHANGE,[s[:-4] for s in SYMS])
    (OUT/"coverage.json").write_text(json.dumps(coverage,indent=2))
r=p.pct_change().fillna(0); cut=int(len(p)*.7)
eda={"start":str(p.index[0]),"end":str(p.index[-1]),"bars":len(p),"annual_vol":(r.std()*np.sqrt(8760)).to_dict(),"correlation":r.corr().to_dict(),"funding_annualized":(f.mean()*8760).to_dict()}
(OUT/"eda.json").write_text(json.dumps(eda,indent=2))
rows=[]; curves={}
def test(name,w,fee,params,extra=None):
    w=w.fillna(0); held=w.shift(2).fillna(0)
    # signal at close, execute next close, earn following close-to-close return
    net=(held*r-held*f).sum(axis=1)-held.diff().abs().sum(axis=1)*fee
    if extra is not None: net=extra
    for label,part in [("train",net.iloc[:cut]),("test",net.iloc[cut:])]:
        rows.append(dict(strategy=name,fee=fee,params=str(params),sample=label,**metrics(part)))
    curves[(name,fee,str(params))]=net
for fee in [.0002,.0005,.001]:
    for lb in [24,72,168]:
        m=p.pct_change(lb); ranks=m.rank(axis=1,method="first")
        test("momentum",(ranks.eq(5).astype(float)-ranks.eq(1).astype(float))/2,fee,lb)
        z=(p-p.rolling(lb).mean())/p.rolling(lb).std()
        for entry in [1.5,2,2.5]:
            test("mean_reversion",(-np.sign(z)*(z.abs()>entry))/5,fee,(lb,entry))
        for a,b in combinations(SYMS,2):
            spread=np.log(p[a])-np.log(p[b]); zz=(spread-spread.rolling(lb).mean())/spread.rolling(lb).std()
            w=p*0; w[a]=-np.sign(zz)*(zz.abs()>2)/2; w[b]=-w[a]
            test("pairs_proxy",w,fee,(a,b,lb))
        # continuous inventory approximation, not a limit-order grid
        test("grid_inventory_proxy",-z.clip(-3,3)/15,fee,lb)
        # delta-neutral carry approximation: spot/perp basis and spot fees not modeled
        score=f.rolling(lb).sum(); w=score.eq(score.max(axis=1),axis=0)&(score>0)
        w=w.div(w.sum(axis=1).replace(0,np.nan),axis=0).fillna(0)*.5
        h=w.shift(2).fillna(0)
        test("funding_proxy",w,fee,lb,(h*f).sum(axis=1)-h.diff().abs().sum(axis=1)*2*fee)
    w=p*0; w["BTCUSDT"]=1
    test("btc_perp",w,fee,"hold")
    test("cash",p*0,fee,"hold")
df=pd.DataFrame(rows)
for fee in [.0002,.0005,.001]:
    train=df[(df["sample"]=="train")&(df.fee==fee)&~df.strategy.isin(["cash","btc_perp"])]
    best=train.sort_values("sharpe",ascending=False).groupby("strategy").head(1)
    net=sum(curves[(x.strategy,x.fee,x.params)] for x in best.itertuples())/len(best)
    test("ensemble",p*0,fee,"train-selected equal",net)
    for target in [.0001,.00025,.0005,.001]:
        d=(1+net.iloc[cut:]).resample("1D").prod()-1
        equity=100000.; paid=0; success=0
        for ret in d:
            profit=equity*ret; equity+=profit
            withdrawal=min(max(profit,0),100000*target)
            equity-=withdrawal; paid+=withdrawal; success+=withdrawal>=100000*target-1e-8
        rows.append(dict(strategy="withdrawal_ensemble",fee=fee,params=str(target),sample="test",days=len(d),ending_equity=equity,withdrawn=paid,withdrawal_success=success/len(d)))
pd.DataFrame(rows).to_csv(OUT/"metrics.csv",index=False)
selected={}
for fee in [.0002,.0005,.001]:
    for x in df[(df["sample"]=="train")&(df.fee==fee)].sort_values("sharpe",ascending=False).groupby("strategy").head(1).itertuples():
        selected[f"{x.strategy}_{fee}"]=curves[(x.strategy,x.fee,x.params)].iloc[cut:]
    selected[f"ensemble_{fee}"]=curves[("ensemble",fee,"train-selected equal")].iloc[cut:]
daily=(1+pd.DataFrame(selected)).resample("1D").prod()-1
daily=daily.iloc[1:-1]
daily.to_csv(OUT/"oos_daily_returns.csv")
daily.corr().to_csv(OUT/"strategy_correlation.csv")
print(pd.DataFrame(rows).query("sample=='test'").sort_values("sharpe",ascending=False).head(20).to_string(index=False))
