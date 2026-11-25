import numpy as np
import pandas as pd

def audit(D,out):
    rows=[]; logs=[]; curves={}
    for fee in D.columns.get_level_values("fee").unique():
        X=D.loc[:,D.columns.get_level_values("fee")==fee]
        for window in [90,180,365,730]:
            for selector in ["sharpe","robust","calmar"]:
                for target in [0,.10,.20]:
                    pieces=[]
                    for start in pd.date_range(X.index.min()+pd.Timedelta(days=window),X.index.max(),freq="MS"):
                        tr=X.loc[(X.index>=start-pd.Timedelta(days=window))&(X.index<start)]
                        te=X.loc[(X.index>=start)&(X.index<start+pd.offsets.MonthBegin(1))]
                        if len(tr)<window*.8 or te.empty: continue
                        sharpe=tr.mean()/tr.std().replace(0,np.nan)*np.sqrt(365)
                        score=sharpe.copy()
                        if selector=="calmar":
                            eq=(1+tr).cumprod(); dd=(eq/eq.cummax().clip(lower=1)-1).min()
                            score=(eq.iloc[-1]**(365/len(tr))-1)/(-dd).replace(0,np.nan)
                        if selector=="robust":
                            for mode in X.columns.get_level_values("mode").unique():
                                g=sharpe.xs((mode,fee),level=("mode","fee")).unstack("rb").sort_index().sort_index(axis=1)
                                for i,lb in enumerate(g.index):
                                    for j,rb in enumerate(g.columns):
                                        a=g.iloc[max(0,i-1):i+2,max(0,j-1):j+2].to_numpy().ravel()
                                        a=a[np.isfinite(a)]
                                        score.loc[(lb,rb,mode,fee)]=max(0,np.median(a))*np.mean(a>0) if len(a) else 0
                        score=score.replace([np.inf,-np.inf],np.nan).dropna()
                        if score.empty or score.max()<=0:
                            pieces.append(pd.Series(0.,index=te.index)); continue
                        key=score.idxmax()
                        scale=min(1,target/(tr[key].std()*np.sqrt(365))) if target and tr[key].std()>0 else 1
                        d=te[key]*scale
                        # Conservative additional monthly transition charge, not exact execution.
                        d=d.copy(); d.iloc[0]-=2*fee
                        pieces.append(d)
                        logs.append(dict(fee=fee,window=window,selector=selector,target=target,date=start,parameter=str(key),scale=scale))
                    if not pieces: continue
                    d=pd.concat(pieces); eq=(1+d).cumprod()
                    key=(fee,window,selector,target); curves[str(key)]=d
                    row=dict(fee=fee,window=window,selector=selector,target=target,days=len(d),cagr=eq.iloc[-1]**(365/len(d))-1,sharpe=d.mean()/d.std()*np.sqrt(365) if d.std()>0 else 0,dd=(eq/eq.cummax().clip(lower=1)-1).min(),positive_days=(d>0).mean(),active_days=(d!=0).mean())
                    for rate in [.0001,.00025,.0005]:
                        capital=1.; paid=0.; success=0
                        for ret in d:
                            profit=capital*ret; capital+=profit
                            draw=min(max(profit,0),rate); capital-=draw; paid+=draw; success+=draw>=rate-1e-12
                        row[f"paid_{rate}"]=paid; row[f"success_{rate}"]=success/len(d); row[f"capital_{rate}"]=capital
                    rows.append(row)
    R=pd.DataFrame(rows); R.to_csv(out/"adaptive_audit.csv",index=False)
    pd.DataFrame(logs).to_csv(out/"adaptive_selection_log.csv",index=False)
    pd.DataFrame(curves).to_csv(out/"adaptive_daily.csv.gz",compression="gzip")
    print("ADAPTIVE AUDIT",R.to_string(index=False),flush=True)
