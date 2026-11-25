import numpy as np
import pandas as pd

def run_breadth(p,r,cut,test,out):
    n=p.shape[1]
    def norm(w):
        return w.div(w.abs().sum(axis=1).replace(0,np.nan),axis=0).fillna(0)
    def basket(mask):
        return norm(mask.astype(float))
    def gate(w,g):
        return w.mul(g.astype(float),axis=0)
    def reb(w,k):
        return w.iloc[::k].reindex(p.index).ffill().fillna(0)
    market=r.mean(axis=1)
    loo=(r.sum(axis=1).values[:,None]-r)/(n-1)
    eda=[]
    for lb in [24,72,168]:
        mom=p.pct_change(lb)
        breadth=mom.gt(0).sum(axis=1)/mom.notna().sum(axis=1).replace(0,np.nan)
        dispersion=mom.std(axis=1)
        # Rolling percentile thresholds adapt to recent market conditions.
        qlo=breadth.rolling(720,min_periods=168).quantile(.25).shift(1)
        qhi=breadth.rolling(720,min_periods=168).quantile(.75).shift(1)
        bull=(breadth>qhi)&(breadth>.5)
        bear=(breadth<qlo)&(breadth<.5)
        highdisp=dispersion>dispersion.rolling(720,min_periods=168).median().shift(1)
        rank=mom.rank(axis=1,pct=True)
        top=basket(rank>.6); bottom=basket(rank<=.4)
        beta=r.rolling(lb).cov(loo)/loo.rolling(lb).var().replace(0,np.nan)
        beta=beta.shift(1).clip(-5,5)
        alpha=(r.rolling(lb).mean()-beta*loo.rolling(lb).mean()).shift(1)
        residual=r-alpha-beta*loo
        accum=residual.rolling(lb).sum()
        z=(accum-accum.rolling(lb).mean().shift(1))/accum.rolling(lb).std().shift(1)
        z=z.replace([np.inf,-np.inf],np.nan)
        # Average pairwise correlation, computed from trailing standardized returns.
        std=r.rolling(lb).std().replace(0,np.nan)
        corr=pd.concat([r[a].rolling(lb).corr(r[b]) for i,a in enumerate(p.columns) for b in p.columns[i+1:]],axis=1).mean(axis=1)
        highcorr=corr>corr.rolling(720,min_periods=168).median().shift(1)
        trend=gate(top,bull)-gate(bottom,bear)
        reversal=gate(bottom,bull)-gate(top,bear)
        cross=(top-bottom)/2
        div=(mom.mean(axis=1)>0)&(breadth.diff(lb)<0)
        divup=(mom.mean(axis=1)<0)&(breadth.diff(lb)>0)
        divergence=gate(-top,div)+gate(bottom,divup)
        hi=p.rolling(lb).max().shift(1); lo=p.rolling(lb).min().shift(1)
        breakout=(p>hi).mean(axis=1); breakdown=(p<lo).mean(axis=1)
        rangeproxy=gate(basket(p>hi),breakout>breakdown)-gate(basket(p<lo),breakdown>breakout)
        for entry in [1.5,2.0,2.5]:
            raw=norm((z.lt(-entry)).astype(float)-(z.gt(entry)).astype(float))
            # Each coin signal hedged with its leave-one-out basket.
            hedge=raw*beta.fillna(0)
            rv=norm(raw-(hedge.sum(axis=1).values[:,None]-hedge)/(n-1))
            switch=gate(trend,highcorr)+gate(rv,~highcorr&highdisp)
            adaptive=gate(trend,highcorr&(bull|bear))+gate(rv,~highcorr&highdisp)+gate(cross,highcorr&~(bull|bear)&highdisp)
            strategies={"breadth_trend":trend,"breadth_reversion":reversal,"breadth_cross_momentum":cross,"breadth_residual":rv,"breadth_divergence":divergence,"breadth_corr_switch":switch,"breadth_range_proxy":rangeproxy,"breadth_adaptive":adaptive,"breadth_residual_momentum":-rv}
            for name,w in strategies.items():
                if entry!=2 and name not in ["breadth_residual","breadth_corr_switch","breadth_adaptive","breadth_residual_momentum"]: continue
                for rb in [1,6,24]:
                    for fee in [.0002,.0005,.001]:
                        test(name,reb(w,rb),fee,(lb,entry,rb))
        for h in [1,4,8,24,72]:
            future=p.shift(-h)/p-1
            for sample,sl in [("train",slice(0,max(0,cut-h))),("test",slice(cut,None))]:
                for a in p:
                    d=pd.DataFrame({"breadth":pd.cut(breadth,[0,.2,.4,.6,.8,1],include_lowest=True),"z":pd.cut(z[a],[-np.inf,-2,-1,0,1,2,np.inf]),"ret":future[a]}).iloc[sl].dropna()
                    for (b,zz),v in d.groupby(["breadth","z"],observed=True):
                        eda.append(dict(asset=a,lb=lb,horizon=h,sample=sample,breadth=str(b),z=str(zz),n=len(v),mean=v.ret.mean(),median=v.ret.median(),positive=(v.ret>0).mean()))
    pd.DataFrame(eda).to_csv(out/"breadth_eda.csv",index=False)
