"""Additional fit-budget, tuned SVR and smooth-model physics controls."""
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from threadpoolctl import threadpool_limits
from revision_experiments import DATA,OUT,GROUP,FEATURES,TARGETS,Model,partitions,ape,xspace

def main():
    df=pd.read_csv(DATA/'02_processed_data/ML_dataset_longform.csv');df['group_id']=df.groupby(GROUP).ngroup()
    rows=[];physics=[];importance=[]
    for seed in [42,137,271]:
        for fold in ['random_supplied','grouped_random']:
            p=partitions(df,seed,fold)
            for target in TARGETS:
                def inv(z):return np.expm1(z) if target=='DelP_Pa' else z
                for fit_kind in ['fit','outer_training']:
                    fit=p['fit'] if fit_kind=='fit' else pd.concat([p[k] for k in ['fit','tune','scale','cal']])
                    y=fit[target].to_numpy();z=np.log1p(y) if target=='DelP_Pa' else y
                    m=ExtraTreesRegressor(n_estimators=200,min_samples_leaf=2,random_state=seed,n_jobs=4).fit(xspace(fit),z)
                    pred=inv(m.predict(xspace(p['test'])))
                    rows.append(dict(seed=seed,fold=fold,target=target,model='ExtraTrees_single',fit_kind=fit_kind,n_fit=len(fit),mape=ape(p['test'][target].to_numpy(),pred).mean()))
                    if seed==42 and fit_kind=='fit':
                        importance.extend(dict(fold=fold,target=target,feature=f,importance=v) for f,v in zip(FEATURES,m.feature_importances_))
                fit=p['fit'];y=fit[target].to_numpy();z=np.log1p(y) if target=='DelP_Pa' else y
                candidates=[]
                center=z.mean();scale=max(z.std(),1e-9);normalized=(z-center)/scale
                for c in [1.,10.,100.]:
                    for gamma in ['scale',.01,.1]:
                        m=make_pipeline(StandardScaler(),SVR(C=c,gamma=gamma,epsilon=.01,tol=.01,max_iter=500000))
                        m.fit(xspace(fit),normalized);loss=ape(p['tune'][target].to_numpy(),inv(m.predict(xspace(p['tune']))*scale+center)).mean()
                        if m[-1].fit_status_==0:candidates.append((loss,m,c,gamma))
                if not candidates:raise RuntimeError('No converged SVR candidate')
                loss,m,c,gamma=min(candidates,key=lambda t:t[0])
                rows.append(dict(seed=seed,fold=fold,target=target,model='SVR_tuned',fit_kind='fit',n_fit=len(fit),mape=ape(p['test'][target].to_numpy(),inv(m.predict(xspace(p['test']))*scale+center)).mean(),C=c,gamma=gamma,tune_mape=loss,fit_status=m[-1].fit_status_))
                print('Control',seed,fold,target,'SVR tune MAPE',loss,flush=True)
        # Smooth model has no tree-range constraint; check local slopes in Re/Pr.
        if seed==42:
            for fold in ['grouped_random','Re100','Re250','Re500']:
                p=partitions(df,seed,fold)
                for target in TARGETS:
                    m=Model('LogPolynomial',target,seed).fit(p['fit'],p['tune'])
                    d=p['test'];base=m.predict(d)[0]
                    for feature in ['Re','Pr']:
                        lo=d.copy();hi=d.copy();lo[feature]=lo[feature]*.99;hi[feature]=hi[feature]*1.01
                        a=m.predict(lo)[0];b=m.predict(hi)[0]
                        exponent=np.log(np.maximum(b,1e-12)/np.maximum(a,1e-12))/np.log(1.01/.99)
                        physics.append(dict(fold=fold,target=target,model='LogPolynomial',feature=feature,negative_slope_fraction=(exponent< -1e-6).mean(),median_exponent=np.median(exponent),min_exponent=exponent.min(),max_exponent=exponent.max()))
    pd.DataFrame(rows).to_csv(OUT/'fit_budget_svr_controls.csv',index=False)
    pd.DataFrame(physics).to_csv(OUT/'local_scaling_physics.csv',index=False)
    pd.DataFrame(importance).to_csv(OUT/'feature_importance.csv',index=False)
    print('Controls complete',flush=True)

if __name__=='__main__':
    with threadpool_limits(limits=2):main()
