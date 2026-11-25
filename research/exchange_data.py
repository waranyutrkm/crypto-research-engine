import ccxt
import pandas as pd
import time

def fetch_exchange(exchange_id, assets, days=180):
    ex=getattr(ccxt,exchange_id)({"enableRateLimit":True})
    ex.load_markets()
    end=int(pd.Timestamp.now(tz="UTC").floor("h").timestamp()*1000)
    start=end-days*86400000
    prices={}; funding={}; coverage={}
    for asset in assets:
        candidates=[m for m in ex.markets.values() if m.get("base")==asset and m.get("swap") and m.get("linear") and m.get("quote") in ("USDT","USDC") and m.get("active") is not False]
        if not candidates: raise ValueError(f"No linear swap: {exchange_id} {asset}")
        m=sorted(candidates,key=lambda m:m["quote"]!="USDT")[0]; symbol=m["symbol"]
        rows=[]; cursor=start
        while cursor<end:
            batch=ex.fetch_ohlcv(symbol,"1h",since=cursor,limit=200)
            if not batch: break
            rows.extend(batch); nxt=max(x[0] for x in batch)+3600000
            if nxt<=cursor: break
            cursor=nxt
        d=pd.DataFrame(rows,columns=["ts","open","high","low","close","volume"]).drop_duplicates("ts").sort_values("ts")
        d=d[(d.ts>=start)&(d.ts<end)]
        if d.empty: raise ValueError(f"No candles {symbol}")
        d.index=pd.to_datetime(d.ts,unit="ms",utc=True)
        prices[asset+"USDT"]=d.close.astype(float)
        if not ex.has.get("fetchFundingRateHistory"): raise ValueError("Funding history unavailable")
        rows=[]; cursor=int(d.ts.min())
        while cursor<end:
            batch=ex.fetch_funding_rate_history(symbol,since=cursor,limit=100)
            batch=[x for x in batch if x.get("timestamp") is not None]
            if not batch: break
            rows.extend(batch); nxt=max(x["timestamp"] for x in batch)+1
            if nxt<=cursor: break
            cursor=nxt
        f=pd.DataFrame(rows)
        if f.empty: raise ValueError(f"No funding history {symbol}")
        f=f.drop_duplicates("timestamp").sort_values("timestamp")
        f=f[(f.timestamp>=start)&(f.timestamp<end)]
        if f.empty: raise ValueError(f"No matching funding history {symbol}")
        f.index=pd.to_datetime(f.timestamp,unit="ms",utc=True)
        funding[asset+"USDT"]=f.fundingRate.astype(float).resample("1h").sum()
        coverage[asset]={"symbol":symbol,"quote":m["quote"],"bars":len(d),"first":str(d.index.min()),"last":str(d.index.max()),"funding_events":len(f),"funding_first":str(f.index.min()),"funding_last":str(f.index.max())}
    p=pd.concat(prices,axis=1,join="inner").sort_index()
    if len(p)<24*30 or (p.index.to_series().diff().dropna()!=pd.Timedelta(hours=1)).any(): raise ValueError("Insufficient or discontinuous common price history")
    for asset,c in coverage.items():
        if pd.Timestamp(c["funding_first"])>p.index[0]+pd.Timedelta(hours=24) or pd.Timestamp(c["funding_last"])<p.index[-1]-pd.Timedelta(hours=24): raise ValueError(f"Incomplete funding endpoints {asset}")
    f=pd.concat(funding,axis=1).reindex(p.index).fillna(0)
    return p,f,coverage
