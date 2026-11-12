"""Add random-reference metrics to earlier checkpoints without retraining.

The initial running process predates the random-score export addition. This
deterministic upgrade uses the identical independent RNG sequence as the runner.
"""
import json
import numpy as np
import pandas as pd
from revision_experiments import OUT,metrics,tie_curve

def main():
    for p in (OUT/'runs').glob('*/complete.json'):
        meta=json.loads(p.read_text());folder=p.parent
        m=pd.read_csv(folder/'metrics.csv')
        if (m.method=='random').any():continue
        d=pd.read_csv(folder/'predictions.csv');rng=np.random.default_rng(meta['seed']+9221);scores={}
        for key in ['tune','scale','cal','test','id_test']:
            n=meta['counts'][key]
            if n:scores[key]=rng.random(n)
        threshold=float(np.quantile(scores['cal'],.90,method='higher'));rows=[];matched=[];curve=[]
        for role,z in d.groupby('role',sort=False):
            score=scores[role];err=z.ape.to_numpy();tol=5. if meta['target']=='Nuavg' else 10.
            base={k:meta[k] for k in ['seed','fold','target','model']};base.update(role=role,method='random')
            rows.append(dict(**base,threshold=threshold,**metrics(score,err,score<=threshold,tol)))
            c,r=tie_curve(score,err)
            for coverage in [.5,.7,.8,.9,.95]:
                ix=min(len(c)-1,int(np.ceil(coverage*len(c)))-1)
                matched.append(dict(**base,coverage=c[ix],mape=r[ix]))
            d.loc[z.index,'random']=score
        pd.concat([m,pd.DataFrame(rows)],ignore_index=True).to_csv(folder/'metrics.csv',index=False)
        d.to_csv(folder/'predictions.csv',index=False)
        old=pd.read_csv(folder/'matched.csv');pd.concat([old,pd.DataFrame(matched)],ignore_index=True).to_csv(folder/'matched.csv',index=False)
    print('Random-reference checkpoints are current.')

if __name__=='__main__':main()
