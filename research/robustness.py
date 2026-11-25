"""Run after daily_cashflow.py exports hourly net returns."""
import numpy as np
import pandas as pd
from pathlib import Path
import json

def bootstrap(daily, target=.0005, reserve_fraction=.2, paths=10000, horizon=365, block=7, seed=42):
    x=np.asarray(daily,dtype=float)
    if len(x)<block or not np.isfinite(x).all(): raise ValueError("Invalid daily returns")
    rng=np.random.default_rng(seed)
    starts=rng.integers(0,len(x)-block+1,size=(paths,int(np.ceil(horizon/block))))
    sim=x[(starts[:,:,None]+np.arange(block)).reshape(paths,-1)[:,:horizon]]
    equity=np.full(paths,1-reserve_fraction,dtype=float); reserve=np.full(paths,reserve_fraction,dtype=float)
    missed=np.zeros(paths); depleted=np.zeros(paths,dtype=bool)
    for t in range(horizon):
        equity*=np.maximum(0,1+sim[:,t])
        paid=np.minimum(target,reserve+equity)
        from_reserve=np.minimum(reserve,paid)
        reserve-=from_reserve; equity-=paid-from_reserve
        missed+=paid<target-1e-12
        depleted|=reserve<=1e-12
    total=equity+reserve
    return dict(target=target,reserve_fraction=reserve_fraction,paths=paths,horizon=horizon,block=block,
        ending_capital_quantiles=np.quantile(total,[.05,.5,.95]).tolist(),
        probability_capital_below_initial=float(np.mean(total<1)),
        probability_missed_withdrawal=float(np.mean(missed>0)),
        probability_reserve_depleted=float(depleted.mean()),
        mean_payment_success=float(1-missed.mean()/horizon))

def main():
    p=Path("results/oos_daily_returns.csv")
    d=pd.read_csv(p,index_col=0)
    result={c:[bootstrap(d[c].dropna(),target=t,reserve_fraction=r,block=b) for t in [.0001,.00025,.0005,.001] for r in [0,.2] for b in [7,30] if len(d[c].dropna())>=b] for c in d.columns}
    Path("results/bootstrap.json").write_text(json.dumps(result,indent=2))
if __name__=="__main__": main()
