import pandas as pd
import numpy as np
from pathlib import Path

def behavior_eda(p,cut,out):
    rows=[]
    for asset in p:
        s=p[asset]
        prior=s.resample("1D").agg(["min","max"]).shift(1).reindex(s.index,method="ffill")
        loc=(s-prior["min"])/(prior["max"]-prior["min"]).replace(0,np.nan)
        regime=np.sign(s.pct_change(168))
        groups={"hour_utc":pd.Series(s.index.hour,index=s.index),"weekday_utc":pd.Series(s.index.dayofweek,index=s.index),"prior_range":pd.cut(loc,[-np.inf,0,.1,.5,.9,1,np.inf]),"trend_168h":regime}
        for horizon in [1,6,24]:
            future=s.shift(-horizon)/s-1
            for sample,idx in [("train",s.index[:max(0,cut-horizon)]),("test",s.index[cut:])]:
                for factor,g in groups.items():
                    d=pd.DataFrame({"group":g,"ret":future}).loc[idx].dropna()
                    for key,v in d.groupby("group",observed=True):
                        x=v.ret
                        rows.append(dict(asset=asset,sample=sample,horizon=horizon,factor=factor,group=str(key),n=len(x),mean=x.mean(),median=x.median(),positive=(x>0).mean(),std=x.std()))
    pd.DataFrame(rows).to_csv(Path(out)/"behavior_eda.csv",index=False)
