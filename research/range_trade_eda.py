import numpy as np
import pandas as pd

def trade_eda(p,out):
    r=p.pct_change(fill_method=None)
    events=[]; episodes=[]; orders=[]
    for lb in [24,168,480,600,720]:
        hi=p.rolling(lb).max().shift(1); lo=p.rolling(lb).min().shift(1)
        valid=hi.notna()&lo.notna()&p.notna()
        up=(p>hi)&valid; dn=(p<lo)&valid
        n=valid.sum(axis=1); net=(up.sum(axis=1)-dn.sum(axis=1))/n.replace(0,np.nan)
        vol=r.rolling(168).std().mean(axis=1)
        for a in p:
            for side,mask,level in [(1,up[a],hi[a]),(-1,dn[a],lo[a])]:
                for t in np.flatnonzero((mask&~mask.shift(1,fill_value=False)&n.ge(4)).values):
                    for h in [1,6,12,24,48,72,168]:
                        path=p[a].iloc[t+1:t+h+1]
                        if len(path)!=h or path.isna().any(): continue
                        x=side*(path/p[a].iloc[t]-1)
                        events.append(dict(time=p.index[t],asset=a,lb=lb,side=side,horizon=h,breadth=net.iloc[t],vol=vol.iloc[t],forward=x.iloc[-1],mfe=max(0,x.max()),mae=min(0,x.min()),false_breakout=bool((side*(path-level.iloc[t])<0).any())))
        u=up.div(up.sum(axis=1).replace(0,np.nan),axis=0).fillna(0)
        v=dn.div(dn.sum(axis=1).replace(0,np.nan),axis=0).fillna(0)
        base=(u.mul(net.gt(0),axis=0)-v.mul(net.lt(0),axis=0)).mul(n.ge(4),axis=0)
        for rb in [6,12,16,24]:
            for offset in sorted(set([0,rb//2])):
                w=base.iloc[offset::rb].reindex(p.index).ffill().fillna(0)
                held=w.shift(2).fillna(0)
                for a in p:
                    h=held[a]; delta=h.diff().fillna(h)
                    for t in np.flatnonzero(delta.ne(0).values):
                        orders.append(dict(time=p.index[t],asset=a,lb=lb,rb=rb,offset=offset,previous_weight=h.iloc[t]-delta.iloc[t],target_weight=h.iloc[t],delta_weight=delta.iloc[t],fee_weight=abs(delta.iloc[t])*.0002))
                    sign=np.sign(h); groups=sign.ne(sign.shift()).cumsum()
                    for _,g in h.groupby(groups):
                        if g.iloc[0]==0: continue
                        start=p.index.get_loc(g.index[0]); end=p.index.get_loc(g.index[-1])
                        if start==0 or end==len(p)-1: continue
                        path=p[a].iloc[start-1:end+1]
                        if path.isna().any(): continue
                        side=np.sign(g.iloc[0]); x=side*(path/path.iloc[0]-1)
                        gross=(g*r[a].loc[g.index]).sum()
                        cost=(delta.loc[g.index].abs().sum()+abs(g.iloc[-1]))*.0002
                        episodes.append(dict(entry=p.index[start-1],exit=p.index[end],asset=a,lb=lb,rb=rb,offset=offset,side=side,hours=len(g),price_return=x.iloc[-1],mfe=x.max(),mae=x.min(),weighted_gross=gross,approx_net=gross-cost))
    E=pd.DataFrame(events); T=pd.DataFrame(episodes); O=pd.DataFrame(orders)
    E.to_csv(out/"breakout_events.csv.gz",index=False,compression="gzip")
    T.to_csv(out/"position_episodes.csv.gz",index=False,compression="gzip")
    O.to_csv(out/"target_weight_changes.csv.gz",index=False,compression="gzip")
    if not E.empty:
        E["breadth_bin"]=pd.cut(E.breadth.abs(),[0,.2,.4,.6,.8,1],include_lowest=True)
        E["hour"]=E.time.dt.hour
        for factor in ["breadth_bin","hour","asset"]:
            g=E.groupby(["lb","side","horizon",factor],observed=True).agg(n=("forward","size"),mean=("forward","mean"),median=("forward","median"),positive=("forward",lambda x:(x>0).mean()),mfe=("mfe","mean"),mae=("mae","mean"),false_rate=("false_breakout","mean"))
            g.to_csv(out/f"event_by_{factor}.csv")
    if not T.empty:
        g=T.groupby(["lb","rb","offset","side"]).agg(n=("approx_net","size"),expectancy=("approx_net","mean"),win_rate=("approx_net",lambda x:(x>0).mean()),holding_hours=("hours","median"),mfe=("mfe","mean"),mae=("mae","mean"))
        g.to_csv(out/"episode_summary.csv")
        print("EPISODE EDA",g.to_string(),flush=True)
    print("EDA COUNTS",len(E),len(T),len(O),flush=True)
