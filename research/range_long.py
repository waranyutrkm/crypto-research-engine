import pandas as pd
import numpy as np
import requests
from io import BytesIO
from zipfile import ZipFile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import time
OUT=Path("range_results"); OUT.mkdir(exist_ok=True)
def fetch(task):
    sym,month=task
    url=f"https://data.binance.vision/data/spot/monthly/klines/{sym}/1h/{sym}-1h-{month}.zip"
    for attempt in range(3):
        try:
            q=requests.get(url,timeout=60)
            if q.status_code==404: return None
            q.raise_for_status()
            with ZipFile(BytesIO(q.content)) as z:
                d=pd.read_csv(z.open(z.namelist()[0]),header=None,usecols=[0,4])
            t=pd.to_numeric(d[0],errors="coerce")
            d=d.loc[t.notna()]
            t=t.dropna().astype("int64")
            t=t.where(t<10**15,t//1000)
            return pd.Series(pd.to_numeric(d[4],errors="raise").values,index=pd.to_datetime(t,unit="ms",utc=True),name=sym)
        except (requests.RequestException,OSError):
            if attempt==2: raise
            time.sleep(2**attempt)
def metrics(d):
    e=(1+d).cumprod()
    return dict(days=len(d),total_return=e.iloc[-1]-1,cagr=e.iloc[-1]**(365/len(d))-1,sharpe=d.mean()/d.std()*np.sqrt(365) if d.std()>0 else 0,max_dd=(e/e.cummax().clip(lower=1)-1).min(),positive_days=(d>0).mean())
symbols=["BTCUSDT","ETHUSDT","BNBUSDT","XRPUSDT","SOLUSDT"]
months=pd.period_range("2018-01",pd.Timestamp.now(tz="UTC").tz_localize(None).to_period("M")-1,freq="M")
series={}
with ThreadPoolExecutor(max_workers=8) as pool:
    for s in symbols:
        parts=[x for x in pool.map(fetch,[(s,str(m)) for m in months]) if x is not None]
        series[s]=pd.concat(parts).sort_index()
p=pd.concat(series,axis=1).sort_index().asfreq("1h")
# No forward fill; a coin is eligible only with complete trailing history.
r=p.pct_change(fill_method=None)
rows=[]; daily={}
for lb in [6,12,18,24,36,48,60,72,96,120,144,168,192,240,288,336,480,600,720]:
    hi=p.rolling(lb).max().shift(1); lo=p.rolling(lb).min().shift(1)
    valid=hi.notna()&lo.notna()&p.notna()
    up=(p>hi)&valid; dn=(p<lo)&valid
    n=valid.sum(axis=1); net=(up.sum(axis=1)-dn.sum(axis=1))/n.replace(0,np.nan)
    for rb in [1,2,3,4,6,8,12,16,24,36,48]:
        for mode in ["majority","strong20","strong40","strong60","scaled","scaled_squared","long_only","short_only"]:
            u=up.div(up.sum(axis=1).replace(0,np.nan),axis=0).fillna(0)
            v=dn.div(dn.sum(axis=1).replace(0,np.nan),axis=0).fillna(0)
            w=u.mul(net.gt(0),axis=0)-v.mul(net.lt(0),axis=0)
            if mode.startswith("strong"): w=w.mul(net.abs().ge(int(mode[6:])/100),axis=0)
            if mode=="scaled": w=w.mul(net.abs(),axis=0)
            if mode=="scaled_squared": w=w.mul(net.abs()**2,axis=0)
            if mode=="short_only": w=w.clip(upper=0)
            if mode=="long_only": w=w.clip(lower=0)
            w=w.mul(n.ge(4),axis=0).iloc[::rb].reindex(p.index).ffill().fillna(0)
            held=w.shift(2).fillna(0)
            missing=(held.ne(0)&r.isna()).any(axis=1)
            # Exclude days with missing held-asset returns from every comparison later.
            for fee in [.0002,.0005,.001]:
                ret=(held*r.fillna(0)).sum(axis=1)-held.diff().abs().sum(axis=1).fillna(0)*fee
                d=(1+ret).resample("1D").prod()-1
                d=d.mask(missing.resample("1D").max())
                daily[(lb,rb,mode,fee)]=d
D=pd.DataFrame(daily).iloc[31:-1].dropna()
D.columns=pd.MultiIndex.from_tuples(D.columns,names=["lb","rb","mode","fee"])
D.to_csv(OUT/"daily_returns.csv.gz",compression="gzip")
# Fixed chronological split; surface selection uses train only.
cut=int(len(D)*.7)
surface=[]
for key in D:
    for sample,g in [("train",D[key].iloc[:cut]),("test",D[key].iloc[cut:])]:
        surface.append(dict(lb=key[0],rb=key[1],mode=key[2],fee=key[3],sample=sample,**metrics(g)))
S=pd.DataFrame(surface)
S.to_csv(OUT/"surface.csv",index=False)
train=S[S["sample"]=="train"].copy()
# Neighborhood is adjacent grid cells in lookback and rebalance, same mode and fee.
for (mode,fee),g in train.groupby(["mode","fee"]):
    grid=g.pivot(index="lb",columns="rb",values="sharpe").sort_index().sort_index(axis=1)
    for i,lb in enumerate(grid.index):
        for j,rb in enumerate(grid.columns):
            a=grid.iloc[max(0,i-1):i+2,max(0,j-1):j+2].to_numpy().ravel()
            mask=(train["mode"]==mode)&(train.fee==fee)&(train.lb==lb)&(train.rb==rb)
            train.loc[mask,"neighbor_median"]=np.median(a)
            train.loc[mask,"neighbor_positive"]=np.mean(a>0)
            train.loc[mask,"neighbor_count"]=len(a)
train["robust_score"]=train.neighbor_median*train.neighbor_positive
train.loc[train.neighbor_median<=0,"robust_score"]=0
ranked=train.merge(S[S["sample"]=="test"],on=["lb","rb","mode","fee"],suffixes=("_train","_test"))
ranked.to_csv(OUT/"robust_surface.csv",index=False)
print("ROBUST TRAIN SELECTED",ranked.sort_values("robust_score",ascending=False).groupby("fee").head(10).to_string(index=False),flush=True)
print("SURFACE DISTRIBUTION",S.groupby(["sample","fee","mode"]).agg(n=("sharpe","size"),median_sharpe=("sharpe","median"),positive_fraction=("cagr",lambda x:(x>0).mean()),median_cagr=("cagr","median")).to_string(),flush=True)
for key in D:
    for year,g in D[key].groupby(D.index.year):
        rows.append(dict(lb=key[0],rb=key[1],mode=key[2],fee=key[3],year=year,**metrics(g)))
pd.DataFrame(rows).to_csv(OUT/"annual.csv",index=False)
for fee in [.0002,.0005,.001]:
    subset=D.loc[:,D.columns.get_level_values("fee")==fee]
    fixed=subset[(24,6,"majority",fee)]
    print("FIXED 24H/6H",fee,metrics(fixed),flush=True)
    print("FIXED ANNUAL",fee,[(int(y),metrics(g)) for y,g in fixed.groupby(fixed.index.year)],flush=True)
    wf=[]; picks=[]
    for start in pd.date_range(D.index.min()+pd.Timedelta(days=365),D.index.max(),freq="MS"):
        train=subset.loc[(subset.index>=start-pd.Timedelta(days=365))&(subset.index<start)]
        test=subset.loc[(subset.index>=start)&(subset.index<start+pd.offsets.MonthBegin(1))]
        if len(train)<300 or test.empty: continue
        scores=train.mean()/train.std().replace(0,np.nan)
        best=scores.idxmax()
        wf.append(test[best] if scores.max()>0 else test[best]*0)
        picks.append(dict(date=str(start),parameter=str(best),train_sharpe=scores.max()*np.sqrt(365)))
    if wf:
        result=pd.concat(wf)
        result.to_csv(OUT/f"walkforward_{fee}.csv")
        pd.DataFrame(picks).to_csv(OUT/f"picks_{fee}.csv",index=False)
        print("WALKFORWARD",fee,metrics(result),flush=True)
print("COVERAGE",D.index.min(),D.index.max(),len(D),flush=True)

from range_trade_eda import trade_eda
trade_eda(p,OUT)

from adaptive_audit import audit
audit(D,OUT)
