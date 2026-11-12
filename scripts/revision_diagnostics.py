"""Derive scientific diagnostics from completed wavy-channel checkpoints."""
from __future__ import annotations
import json
import numpy as np
import pandas as pd
from revision_experiments import OUT,DATA,GROUP,metrics,aggregate,tie_curve

def cluster_intervals(d,tolerance,rng,reps=500,block='group_id'):
    # Resample non-Pr tuples, preserving all Pr rows in each selected tuple.
    groups=[g.index.to_numpy() for _,g in d.reset_index(drop=True).groupby(block)]
    if len(groups)<3:
        return dict(n_clusters=len(groups),status='too_few_clusters',**{name+suffix:np.nan for name in ['all_mape','coverage','retained_mape','retained_failure'] for suffix in ['_lo','_hi']})
    e=d.ape.to_numpy(); a=d.accepted.to_numpy(bool); score=d.UD_max.to_numpy()
    vals=[]
    for _ in range(reps):
        ix=np.concatenate([groups[j] for j in rng.integers(0,len(groups),len(groups))])
        k=a[ix];err=e[ix]
        vals.append([err.mean(),k.mean(),err[k].mean() if k.any() else np.nan,(err[k]>tolerance).mean() if k.any() else np.nan])
    vals=np.asarray(vals); result={}
    for j,name in enumerate(['all_mape','coverage','retained_mape','retained_failure']):
        x=vals[:,j];x=x[np.isfinite(x)]
        result[name+'_lo']=float(np.quantile(x,.025)) if len(x) else np.nan
        result[name+'_hi']=float(np.quantile(x,.975)) if len(x) else np.nan
    return dict(n_clusters=len(groups),status='conditional_descriptive',**result)

def main():
    aggregate();m=pd.read_csv(OUT/'metrics.csv');s=pd.read_csv(OUT/'sensitivity.csv')
    keys=['fold','target','model','role','method']
    values=['coverage','all_mape','retained_mape','retained_failure','false_accept','false_abstain','aurc','auroc','auprc','spearman']
    summary=m.groupby(keys)[values].agg(['mean','std','count']);summary.columns=['_'.join(c) for c in summary.columns]
    summary.reset_index().to_csv(OUT/'summary.csv',index=False)
    interval=[];coarse=[];tolrows=[];mixrows=[];diag=[];maps=[];availability=[]
    rng=np.random.default_rng(94371)
    for p in sorted((OUT/'runs').glob('*/complete.json')):
        meta=json.loads(p.read_text());seed=meta['seed'];fold=meta['fold'];target=meta['target'];model=meta['model']
        d=pd.read_csv(p.parent/'predictions.csv');base=dict(seed=seed,fold=fold,target=target,model=model)
        availability.append(dict(**base,learned_gate_status='available' if 'learned' in d.columns else 'unavailable_single_class_tune_failures'))
        tolerance=5. if target=='Nuavg' else 10.
        test=d[d.role=='test'].reset_index(drop=True)
        if seed==42:
            interval.append(dict(**base,**cluster_intervals(test,tolerance,rng)))
            for block in ['geometry_id','Re']:
                coarse.append(dict(**base,block=block,**cluster_intervals(test,tolerance,rng,block=block)))
            for tol in ([1,2,5,10,15] if target=='Nuavg' else [2,5,10,15,20]):
                tolrows.append(dict(**base,tolerance=tol,**metrics(test.UD_max.to_numpy(),test.ape.to_numpy(),test.accepted.to_numpy(bool),tol)))
        if fold not in ['random_supplied','grouped_random']:
            iddata=d[d.role=='id_test'];ood=test
            for fraction in [.0,.25,.5,.75,1.]:
                # Fixed-size mixture; subsampling does not use output labels.
                total=min(400,len(iddata),len(ood));n_ood=round(total*fraction);n_id=total-n_ood
                a=iddata.sample(n=n_id,random_state=seed).assign(is_ood=0)
                b=ood.sample(n=n_ood,random_state=seed).assign(is_ood=1)
                z=pd.concat([a,b]);score=z.UD_max.to_numpy();err=z.ape.to_numpy();accepted=z.accepted.to_numpy(bool)
                result=metrics(score,err,accepted,tolerance)
                if n_ood and n_id:
                    from sklearn.metrics import roc_auc_score,average_precision_score
                    result['ood_auroc']=roc_auc_score(z.is_ood,score);result['ood_auprc']=average_precision_score(z.is_ood,score)
                mixrows.append(dict(**base,ood_fraction=fraction,**result))
        if seed==42:
            for re,g in test.groupby('Re'):
                diag.append(dict(**base,Re=re,n=len(g),mean_signed_error=g.signed_error_pct.mean(),mean_ape=g.ape.mean(),mean_U=g.raw_U.mean(),mean_knn=g.raw_knn.mean(),mean_maha=g.raw_maha.mean(),mean_gap=g.support_gap.mean(),physics_activation=(g.raw_V>1e-12).mean(),coverage=g.accepted.mean()))
            if fold=='grouped_random' and model in ['ExtraTrees_seed','XGBoost_seed','GP_ARD']:
                for (re,da),g in test.groupby(['Re','Da']):
                    maps.append(dict(**base,Re=re,Da=da,score=g.UD_max.mean(),coverage=g.accepted.mean(),n=len(g)))
    for name,rows in [('cluster_intervals',interval),('coarse_cluster_intervals',coarse),('tolerance_sweep',tolrows),('mixed_shift',mixrows),('re_diagnostics',diag),('map_data',maps),('learned_gate_availability',availability)]:pd.DataFrame(rows).to_csv(OUT/f'{name}.csv',index=False)
    # Warn tier audit on main 80/90 thresholds, no risk ordering assumed.
    w=s[(s.q_predict==.8)&(s.q_warn==.9)].pivot_table(index=['seed','fold','target','model','role'],columns='state',values='mape').reset_index()
    w['warn_lower_than_predict']=w.warn<w.predict
    w.to_csv(OUT/'warning_order_audit.csv',index=False)
    audit=pd.read_csv(OUT/'pressure_pr_audit.csv')
    details=dict(nonconstant_groups=int((audit['nunique']>1).sum()),
        above_1e_6_percent=int((audit.relative_range_pct>1e-6).sum()),max_relative_range_pct=audit.relative_range_pct.max(),
        relative_range_quantiles=audit.relative_range_pct.quantile([.5,.9,.95,1.]).to_dict())
    (OUT/'pressure_pr_details.json').write_text(json.dumps(details,indent=2),encoding='utf-8')
    print('Diagnostics complete:',len(m),'metric rows;',len(interval),'cluster intervals;',len(mixrows),'mixtures.')

if __name__=='__main__':main()
