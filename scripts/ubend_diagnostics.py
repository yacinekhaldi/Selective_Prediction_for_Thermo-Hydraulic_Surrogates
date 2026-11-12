"""Derive U-bend diagnostics from completed checkpoints without model selection."""
import json
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import roc_auc_score,average_precision_score
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
import ubend_experiments as U

def bootstrap(d,tol,rng,block='group_id',reps=500):
    groups=[g.index.to_numpy() for _,g in d.reset_index(drop=True).groupby(block)]
    n=len(groups)
    fields=['all_mape','coverage','retained_mape','retained_failure']
    if n<3:return dict(n_clusters=n,status='too_few_clusters',**{f+s:np.nan for f in fields for s in ['_lo','_hi']})
    # Cluster sufficient statistics preserve all observations in a sampled block.
    err=d.ape.to_numpy();accepted=d.accepted.to_numpy(bool)
    totals=np.array([[len(g),err[g].sum(),accepted[g].sum(),err[g][accepted[g]].sum(),((err[g]>tol)&accepted[g]).sum()] for g in groups])
    summed=totals[rng.integers(0,n,size=(reps,n))].sum(axis=1)
    size,loss,kept,retained_loss,fail=summed.T
    risk=np.divide(retained_loss,kept,out=np.full(reps,np.nan),where=kept>0)
    failure=np.divide(fail,kept,out=np.full(reps,np.nan),where=kept>0)
    result=dict(n_clusters=n,status='conditional_descriptive',sampling_unit=block,replicates=reps)
    for name,values in zip(fields,[loss/size,kept/size,risk,failure]):
        valid=values[np.isfinite(values)]
        result[name+'_lo']=np.quantile(valid,.025) if len(valid) else np.nan
        result[name+'_hi']=np.quantile(valid,.975) if len(valid) else np.nan
    return result

def main():
    U.aggregate()
    m=pd.read_csv(U.OUT/'metrics.csv');s=pd.read_csv(U.OUT/'sensitivity.csv')
    keys=['fold','target','model','role','method']
    values=['coverage','all_mape','retained_mape','retained_failure','false_accept','false_abstain','failure_prevalence','aurc','auroc','auprc','spearman']
    summary=m.groupby(keys)[values].agg(['mean','std','count']);summary.columns=['_'.join(c) for c in summary.columns]
    summary.reset_index().to_csv(U.OUT/'summary.csv',index=False)
    full=pd.read_csv(U.DATA/'ubend_longform.csv')
    x=StandardScaler().fit_transform(full[U.FEATURES])
    clusters=KMeans(n_clusters=6,random_state=42,n_init=10).fit_predict(x)
    mapping=full[['case_uid','sample_id','geometry_id']+U.FEATURES].copy();mapping['geometry_cluster']=clusters
    mapping['input_0_bin']=pd.qcut(full.input_0,5,labels=False)
    mapping['input_6_bin']=pd.qcut(full.input_6,5,labels=False)
    mapping.to_csv(U.OUT/'geometry_mapping.csv',index=False)
    mapping=mapping.set_index('case_uid')
    rng=np.random.default_rng(94371)
    interval=[];coarse=[];tolrows=[];mixrows=[];diag=[];maps=[];availability=[];binary=[];cqr=[];physics=[];distributions=[]
    for p in sorted((U.OUT/'runs').glob('*/complete.json')):
        meta=json.loads(p.read_text());seed=meta['seed'];fold=meta['fold'];target=meta['target'];model=meta['model']
        d=pd.read_csv(p.parent/'predictions.csv');base=dict(seed=seed,fold=fold,target=target,model=model)
        availability.append(dict(**base,learned_gate_status='available' if 'learned' in d.columns else 'unavailable_single_class_tune_failures'))
        tol=5. if target=='CoolingObjective_K2m2' else 10.
        test=d[d.role=='test'].reset_index(drop=True)
        for col in ['geometry_cluster','input_0_bin','input_6_bin']:test[col]=mapping.loc[test.case_uid,col].to_numpy()
        if seed==42:
            interval.append(dict(**base,**bootstrap(test,tol,rng)))
            for col in ['geometry_cluster','input_0_bin']:
                coarse.append(dict(**base,block=col,**bootstrap(test,tol,rng,block=col)))
            for tolerance in ([1,2,5,10,15] if target=='CoolingObjective_K2m2' else [2,5,10,15,20]):
                tolrows.append(dict(**base,tolerance=tolerance,**U.metrics(test.UD_max.to_numpy(),test.ape.to_numpy(),test.accepted.to_numpy(bool),tolerance)))
        if fold not in ['random_supplied','grouped_random']:
            iddata=d[d.role=='id_test']
            for fraction in [0,.25,.5,.75,1.]:
                total=min(400,len(iddata),len(test));n_ood=round(total*fraction)
                a=iddata.sample(n=total-n_ood,random_state=seed).assign(is_ood=0)
                b=test.sample(n=n_ood,random_state=seed).assign(is_ood=1)
                z=pd.concat([a,b]);score=z.UD_max.to_numpy();err=z.ape.to_numpy();accepted=z.accepted.to_numpy(bool)
                result=U.metrics(score,err,accepted,tol)
                result['ood_auroc']=roc_auc_score(z.is_ood,score) if 0<n_ood<total else np.nan
                result['ood_auprc']=average_precision_score(z.is_ood,score) if 0<n_ood<total else np.nan
                mixrows.append(dict(**base,ood_fraction=fraction,**result))
        if seed==42:
            for region,g in test.groupby('geometry_cluster'):
                diag.append(dict(**base,geometry_cluster=region,n=len(g),mean_signed_error=g.signed_error_pct.mean(),mean_ape=g.ape.mean(),mean_U=g.raw_U.mean(),mean_knn=g.raw_knn.mean(),mean_maha=g.raw_maha.mean(),mean_gap=g.support_gap.mean(),physics_activation=(g.raw_V>1e-12).mean(),coverage=g.accepted.mean()))
            if fold=='grouped_random' and model in ['ExtraTrees_seed','XGBoost_seed','GP_ARD']:
                for (a,b),g in test.groupby(['input_0_bin','input_6_bin']):
                    maps.append(dict(**base,input_0_bin=a,input_6_bin=b,score=g.UD_max.mean(),coverage=g.accepted.mean(),n=len(g)))
        for role,g in d.groupby('role'):
            physics.append(dict(**base,role=role,activation_fraction=(g.raw_V>1e-12).mean(),mean_V=g.raw_V.mean()))
            truth=g.truth.to_numpy();pred=g.prediction.to_numpy();err=g.ape.to_numpy()
            distributions.append(dict(**base,role=role,n=len(g),mape=err.mean(),median_ape=np.median(err),p90_ape=np.quantile(err,.90),p95_ape=np.quantile(err,.95),p99_ape=np.quantile(err,.99),max_ape=err.max(),mean_smape=np.mean(200*abs(pred-truth)/np.maximum(abs(pred)+abs(truth),1e-12)),rmse=np.sqrt(np.mean((pred-truth)**2)),mae=np.mean(abs(pred-truth)),prediction_min=pred.min(),prediction_max=pred.max(),truth_min=truth.min(),truth_max=truth.max(),nonpositive_prediction_fraction=(pred<=0).mean(),logpoly_lower_guard_hits=int(np.isclose(pred,np.exp(-30),rtol=1e-9,atol=0).sum()) if model=='LogPolynomial' else 0,logpoly_upper_guard_hits=int(np.isclose(pred,np.exp(30),rtol=1e-9,atol=0).sum()) if model=='LogPolynomial' else 0))
        # Binary upper-quantile acceptance can be recovered exactly from the
        # diagnostic state counts; independently test every q=.90 record below.
        for (role,q),g in pd.read_csv(p.parent/'sensitivity.csv').query('q_predict == .60').groupby(['role','q_warn']):
            kept=g[g.state.isin(['predict','warn'])];n=int(g.n.sum());nk=int(kept.n.sum())
            binary.append(dict(**base,role=role,calibration_quantile=q,n=n,n_retained=nk,coverage=nk/n,threshold=g.tau_warn.iloc[0],retained_mape=(kept.mape.fillna(0)*kept.n).sum()/nk if nk else np.nan,retained_failure=(kept.failure.fillna(0)*kept.n).sum()/nk if nk else np.nan))
        if model=='CQR':
            ep=pd.read_csv(p.parent/'interval_endpoints.csv')
            for role,z in ep.groupby('role'):
                cqr.append(dict(**base,role=role,rows=len(z),groups=z.group_id.nunique(),row_coverage=z.covered.mean(),group_coverage=z.groupby('group_id').covered.all().mean(),mean_width=(z.upper-z.lower).mean()))
    exports=[('cluster_intervals',interval),('coarse_cluster_intervals',coarse),('tolerance_sweep',tolrows),('mixed_shift',mixrows),('geometry_diagnostics',diag),('map_data',maps),('learned_gate_availability',availability),('binary_coverage_sweep',binary),('conformal_group_coverage',cqr),('physics_activation',physics),('error_distribution',distributions)]
    for name,rows in exports:pd.DataFrame(rows).to_csv(U.OUT/f'{name}.csv',index=False)
    b=pd.DataFrame(binary);primary=m[m.method=='UD_max']
    check=b[b.calibration_quantile==.9].merge(primary,on=['seed','fold','target','model','role'],suffixes=('_binary','_main'),validate='one_to_one')
    assert len(check)==len(primary)
    for col in ['coverage','threshold','retained_mape','retained_failure']:np.testing.assert_allclose(check[col+'_binary'],check[col+'_main'],rtol=1e-10,atol=1e-10,equal_nan=True)
    w=s[(s.q_predict==.8)&(s.q_warn==.9)].pivot_table(index=['seed','fold','target','model','role'],columns='state',values='mape').reset_index()
    if 'warn' not in w:w['warn']=np.nan
    w['warn_lower_than_predict']=w.warn<w.predict;w.to_csv(U.OUT/'warning_order_audit.csv',index=False)
    U.save_json(dict(pressure_pr_audit='not applicable: no Pr variants; each valid geometry is unique',re_diagnostics='not applicable: fixed operating conditions; see geometry_diagnostics.csv',local_scaling_physics='no transferable monotonicity assumption for signed geometry coordinates; see physics_activation.csv',coarse_bootstrap='six covariate-only KMeans strata and five input_0 bins; descriptive conditional blocks, not physical geometry families or unseen-geometry confidence intervals',conformal='one row per design, so group and row coverage coincide; exchangeability is a hypothetical condition, not established by Sobol sampling or imposed shifts',tolerances='5% cooling and 10% pressure are illustrative analogues, not validated engineering requirements'),U.OUT/'not_applicable_and_limits.json')
    print(f'Diagnostics: {len(interval)} design intervals, {len(mixrows)} mixtures, {len(cqr)} CQR coverage records.',flush=True)

if __name__=='__main__':
    with threadpool_limits(limits=3):main()
