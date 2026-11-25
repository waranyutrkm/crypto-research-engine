import numpy as np
import pandas as pd

def factors(p,out):
    r=p.pct_change(fill_method=None)
    cut=int(len(p)*.7)
    rows=[]; lead=[]
    for lb in [24,168,480,600,720]:
        hi=p.rolling(lb).max().shift(1); lo=p.rolling(lb).min().shift(1)
        valid=hi.notna()&lo.notna()&p.notna()
        up=(p>hi)&valid; dn=(p<lo)&valid
        n=valid.sum(axis=1)
        breadth=(up.sum(axis=1)-dn.sum(axis=1))/n.replace(0,np.nan)
        vol=r.rolling(168).std().mean(axis=1)
        volratio=vol/vol.rolling(720,min_periods=168).median().shift(1)
        trend=p.pct_change(168,fill_method=None).mean(axis=1)
        for a in p:
            others=r.drop(columns=a).mean(axis=1)
            for lag in [1,3,6,12,24]:
                leader=(1+others).rolling(lag).apply(np.prod,raw=True)-1
                for h in [1,6,24,72]:
                    future=p[a].shift(-h)/p[a]-1
                    for sample,sl in [("train",slice(0,cut-h)),("test",slice(cut,None))]:
                        d=pd.concat([leader.rename("x"),future.rename("y")],axis=1).iloc[sl].dropna()
                        lead.append(dict(asset=a,lb=lb,lag=lag,horizon=h,sample=sample,n=len(d),correlation=d.x.corr(d.y)))
            for side,mask,level in [(1,up[a],hi[a]),(-1,dn[a],lo[a])]:
                event=mask&~mask.shift(1,fill_value=False)&n.ge(4)
                for h in [1,6,12,24,48,72,168]:
                    future=side*(p[a].shift(-h)/p[a]-1)
                    d=pd.DataFrame({"ret":future,"event":event,"confirmation":pd.cut(side*breadth,[-1,-.2,0,.2,.4,.6,1],include_lowest=True),"volatility":pd.cut(volratio,[0,.75,1,1.5,np.inf]),"trend":np.sign(trend),"hour":p.index.hour,"weekday":p.index.dayofweek})
                    for sample,sl in [("train",slice(0,cut-h)),("test",slice(cut,None))]:
                        z=d.iloc[sl]; z=z[z.event]
                        for factor in ["confirmation","volatility","trend","hour","weekday"]:
                            for key,g in z.dropna(subset=["ret",factor]).groupby(factor,observed=True):
                                rows.append(dict(asset=a,lb=lb,side=side,horizon=h,sample=sample,factor=factor,group=str(key),n=len(g),mean=g.ret.mean(),median=g.ret.median(),positive=(g.ret>0).mean()))
    pd.DataFrame(rows).to_csv(out/"behavior_factor_oos.csv",index=False)
    pd.DataFrame(lead).drop(columns="lb").drop_duplicates().to_csv(out/"lead_lag_oos.csv",index=False)
    print("FACTOR EDA COUNTS",len(rows),len(lead),flush=True)
