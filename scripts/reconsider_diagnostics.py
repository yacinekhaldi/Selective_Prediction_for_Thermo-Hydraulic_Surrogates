"""Audit existing experiments and add diagnostics without changing fitted policies."""
import json
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from revision_experiments import OUT, DATA, GROUP, Model, partitions, xspace, ape, metrics

def main():
    df=pd.read_csv(DATA/'02_processed_data/ML_dataset_longform.csv')
    df['group_id']=df.groupby(GROUP).ngroup()
    extreme=[]; numerical=[]; sweep=[]
    # Refit every G01 log-polynomial, including all three complete seeds.
    for seed in [42,137,271]:
        parts=partitions(df,seed,'G01')
        for target in ['Nuavg','DelP_Pa']:
            model=Model('LogPolynomial',target,seed).fit(parts['fit'],parts['tune'])
            d=parts['test']; z=model.members[0].predict(model.scaler.transform(xspace(d)))
            pred=model.predict(d)[0]
            stored=pd.read_csv(OUT/'runs'/f'{seed}_G01_{target}_LogPolynomial'/'predictions.csv')
            stored=stored[stored.role=='test']
            np.testing.assert_allclose(pred,stored.prediction,rtol=1e-7,atol=1e-10)
            err=ape(d[target].to_numpy(),pred)
            np.testing.assert_allclose(err,stored.ape,rtol=1e-7,atol=1e-5)
            np.testing.assert_allclose(ape(stored.truth.to_numpy(),stored.prediction.to_numpy()),stored.ape,rtol=1e-12,atol=1e-10)
            smape=200*abs(pred-d[target].to_numpy())/(abs(pred)+abs(d[target].to_numpy()))
            extreme.append(dict(seed=seed,target=target,fold='G01',n=len(d),degree=model.meta['degree'],ridge_alpha=model.meta['ridge_alpha'],truth_min=d[target].min(),truth_max=d[target].max(),raw_log_min=z.min(),raw_log_max=z.max(),lower_clipped=int((z < -30).sum()),upper_clipped=int((z > 30).sum()),prediction_min=pred.min(),prediction_max=pred.max(),mape=err.mean(),median_ape=np.median(err),p95_ape=np.quantile(err,.95),mean_smape=smape.mean(),max_ape=err.max()))
    # Verify sklearn restores target units before the pressure inverse transform.
    parts=partitions(df,42,'grouped_random')
    model=Model('GP_ARD','DelP_Pa',42).fit(parts['fit'],parts['tune'])
    gp=model.members[0]; x=model.scaler.transform(xspace(parts['test']))
    m,s=gp.predict(x,return_std=True)
    k=gp.kernel_(x,gp.X_train_)
    manual=k@gp.alpha_*gp._y_train_std+gp._y_train_mean
    np.testing.assert_allclose(m,manual,rtol=1e-10,atol=1e-10)
    from scipy.linalg import solve_triangular
    v=solve_triangular(gp.L_,k.T,lower=True)
    manual_std=np.sqrt(np.maximum(0,gp.kernel_.diag(x)-(v*v).sum(axis=0)))*gp._y_train_std
    np.testing.assert_allclose(s,manual_std,rtol=1e-7,atol=1e-9)
    stored=pd.read_csv(OUT/'runs/42_grouped_random_DelP_Pa_GP_ARD/predictions.csv')
    stored=stored[stored.role=='test']
    np.testing.assert_allclose(model.inverse(m),stored.prediction,rtol=1e-8,atol=1e-8)
    numerical.append(dict(check='GP_pressure_unstandardized_predict',max_mean_difference=float(abs(m-manual).max()),max_std_difference=float(abs(s-manual_std).max()),mean_clip_count=int(((m < -30)|(m > 30)).sum()),variance_clip_count=int(((s*s >30)|(2*m+s*s < -60)|(2*m+s*s >60)).sum()),n=len(m)))
    # Existing two-threshold outputs already contain every needed calibrated
    # cutoff; select predict+warn for a binary upper-quantile sensitivity study.
    sens=pd.read_csv(OUT/'sensitivity.csv')
    for keys,g in sens[sens.q_predict==.60].groupby(['seed','fold','target','model','role','q_warn']):
        kept=g[g.state.isin(['predict','warn'])]; n=int(g.n.sum()); nk=int(kept.n.sum())
        sweep.append(dict(zip(['seed','fold','target','model','role','calibration_quantile'],keys),n=n,n_retained=nk,coverage=nk/n,threshold=float(g.tau_warn.iloc[0]),retained_mape=float((kept.mape.fillna(0)*kept.n).sum()/nk) if nk else np.nan,retained_failure=float((kept.failure.fillna(0)*kept.n).sum()/nk) if nk else np.nan))
    pd.DataFrame(extreme).to_csv(OUT/'logpoly_extreme_audit.csv',index=False)
    binary=pd.DataFrame(sweep)
    primary=pd.read_csv(OUT/'metrics.csv')
    primary=primary[primary.method=='UD_max']
    check=binary[binary.calibration_quantile==.9].merge(primary,on=['seed','fold','target','model','role'],suffixes=('_binary','_primary'),validate='one_to_one')
    assert len(check)==len(primary)
    for column in ['coverage','retained_mape','retained_failure','threshold']:
        np.testing.assert_allclose(check[column+'_binary'],check[column+'_primary'],rtol=1e-10,atol=1e-10,equal_nan=True)
    binary.to_csv(OUT/'binary_coverage_sweep.csv',index=False)
    (OUT/'transform_audit.json').write_text(json.dumps(numerical,indent=2))
    m=pd.read_csv(OUT/'metrics.csv'); matched=pd.read_csv(OUT/'matched.csv')
    print('EXTREME\n'+pd.DataFrame(extreme).to_string(index=False))
    a=matched[(matched.role=='test')&(matched.coverage.between(.899,.902))]
    p=a.pivot_table(index=['model','target','fold'],columns='method',values='mape')
    p['max_minus_U']=p.UD_max-p.U;p['max_minus_D']=p.UD_max-p.D
    print('FUSION HELPS\n'+p.sort_values('max_minus_U').head(4).to_string())
    print('FUSION HURTS\n'+p.sort_values('max_minus_U').tail(4).to_string())
    print('PHYSICS\n'+p.loc[(p.index.get_level_values('model').isin(['GP_ARD','LogPolynomial'])) & (p.UDV_max!=p.UD_max),['UD_max','UDV_max']].head(12).to_string())
    print('DETECTION\n'+m[(m.seed==42)&(m.role=='test')&(m.method=='UD_max')&(m.model=='XGBoost_seed')&(m.target=='Nuavg')&(m.fold.isin(['Re250','Re500','G04']))][['fold','failure_prevalence','auroc','auprc','coverage','retained_failure']].to_string(index=False))
    print('GP\n'+m[(m.role=='test')&(m.method=='UD_max')&(m.model=='GP_ARD')&(m.fold.str.startswith('G'))].groupby(['target','fold'])[['coverage','retained_mape']].mean().to_string())
    print('TRANSFORM\n'+str(numerical))

if __name__=='__main__':
    with threadpool_limits(limits=2):main()


