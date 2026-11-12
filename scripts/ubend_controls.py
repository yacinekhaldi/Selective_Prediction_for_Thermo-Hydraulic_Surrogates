"""Fit-budget, tuned-SVR and original-family point benchmarks for U-bend."""
import hashlib,json,time
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor,RandomForestRegressor,HistGradientBoostingRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
from xgboost import XGBRegressor
from threadpoolctl import threadpool_limits
import ubend_experiments as U

def main():
    df=pd.read_csv(U.DATA/'ubend_longform.csv');df['group_id']=df.sample_id
    checkpoint=U.OUT/'controls';checkpoint.mkdir(exist_ok=True)
    modeldir=U.OUT/'models';modeldir.mkdir(exist_ok=True)
    for seed in U.SEEDS:
        for fold in ['random_supplied','grouped_random']:
            parts=U.partitions(df,seed,fold)
            for target in U.TARGETS:
                folder=checkpoint/f'{seed}_{fold}_{target}';folder.mkdir(exist_ok=True)
                if (folder/'complete.json').exists():continue
                rows=[];benchmarks=[];importance=[];predictions=[];candidates=[]
                def transform(y):return np.log1p(y) if target=='DelP_Pa' else y
                def inverse(y):return np.expm1(np.clip(y,-30,30)) if target=='DelP_Pa' else y
                test=parts['test'];truth=test[target].to_numpy()
                outer=pd.concat([parts[k] for k in ['fit','tune','scale','cal']])
                for kind,fit in [('fit',parts['fit']),('outer_training',outer)]:
                    m=make_pipeline(StandardScaler(),ExtraTreesRegressor(n_estimators=200,min_samples_leaf=2,random_state=seed,n_jobs=3))
                    start=time.perf_counter();m.fit(U.xspace(fit),transform(fit[target].to_numpy()))
                    pred=inverse(m.predict(U.xspace(test)))
                    rows.append(dict(seed=seed,fold=fold,target=target,model='ExtraTrees_single',fit_kind=kind,n_fit=len(fit),mape=U.ape(truth,pred).mean(),fit_seconds=time.perf_counter()-start,fit_case_hash=hashlib.sha256('\n'.join(fit.case_uid).encode()).hexdigest()))
                    if seed==42 and kind=='fit':importance.extend(dict(fold=fold,target=target,feature=f,importance=v) for f,v in zip(U.FEATURES,m[-1].feature_importances_))
                fit=parts['fit'];z=transform(fit[target].to_numpy());center=z.mean();scale=max(z.std(),1e-9)
                best=None;start=time.perf_counter()
                for c in [1.,10.,100.]:
                    for gamma in ['scale',.01,.1]:
                        m=make_pipeline(StandardScaler(),SVR(C=c,gamma=gamma,epsilon=.01,tol=.01,max_iter=500000))
                        m.fit(U.xspace(fit),(z-center)/scale)
                        loss=U.ape(parts['tune'][target].to_numpy(),inverse(m.predict(U.xspace(parts['tune']))*scale+center)).mean()
                        candidates.append(dict(seed=seed,fold=fold,target=target,C=c,gamma=gamma,tune_mape=loss,fit_status=m[-1].fit_status_))
                        if m[-1].fit_status_==0 and (best is None or loss<best[0]):best=(loss,m,c,gamma)
                assert best is not None,'No converged SVR candidate'
                loss,m,c,gamma=best;pred=inverse(m.predict(U.xspace(test))*scale+center)
                rows.append(dict(seed=seed,fold=fold,target=target,model='SVR_tuned',fit_kind='fit',n_fit=len(fit),mape=U.ape(truth,pred).mean(),C=c,gamma=gamma,tune_mape=loss,fit_status=m[-1].fit_status_,fit_seconds=time.perf_counter()-start))
                # Original five baseline families/settings, on full outer training.
                models={
                  'RandomForest':RandomForestRegressor(n_estimators=500,min_samples_leaf=2,random_state=seed,n_jobs=3),
                  'ExtraTrees':ExtraTreesRegressor(n_estimators=500,min_samples_leaf=2,random_state=seed,n_jobs=3),
                  'HistGradientBoosting':HistGradientBoostingRegressor(max_iter=450,learning_rate=.045,random_state=seed),
                  'SVR_RBF':SVR(C=80.,epsilon=.02,gamma='scale',max_iter=500000),
                  'XGBoost':XGBRegressor(n_estimators=800,max_depth=6,learning_rate=.03,subsample=.95,colsample_bytree=.95,random_state=seed,n_jobs=3,tree_method='hist')}
                for name,estimator in models.items():
                    m=make_pipeline(StandardScaler(),estimator);start=time.perf_counter()
                    m.fit(U.xspace(outer),transform(outer[target].to_numpy()))
                    pred=inverse(m.predict(U.xspace(test)))
                    benchmarks.append(dict(seed=seed,fold=fold,target=target,model=name,n_fit=len(outer),n_test=len(test),mape=U.ape(truth,pred).mean(),mae=mean_absolute_error(truth,pred),rmse=np.sqrt(mean_squared_error(truth,pred)),r2=r2_score(truth,pred),fit_seconds=time.perf_counter()-start,fit_status=int(getattr(estimator,'fit_status_',0))))
                    predictions.append(pd.DataFrame(dict(seed=seed,fold=fold,target=target,model=name,case_uid=test.case_uid.to_numpy(),truth=truth,prediction=pred,ape=U.ape(truth,pred))))
                    if seed==42 and fold=='random_supplied':
                        stem=f'{name}_{target}'
                        joblib.dump({'estimator':m,'features':U.FEATURES,'target':target,'target_transform':'log1p' if target=='DelP_Pa' else 'identity','inverse_target_transform':'expm1(clip(-30,30))' if target=='DelP_Pa' else 'identity','seed':seed,'fold':fold},modeldir/(stem+'.joblib'),compress=3)
                        U.save_json(dict(model=name,target=target,seed=seed,fold=fold,n_fit=len(outer),features=U.FEATURES,settings=estimator.get_params(),source=U.audit.__module__),modeldir/(stem+'.json'))
                pd.DataFrame(rows).to_csv(folder/'fit_budget.csv',index=False)
                pd.DataFrame(benchmarks).to_csv(folder/'point_baselines.csv',index=False)
                pd.DataFrame(importance,columns=['fold','target','feature','importance']).to_csv(folder/'feature_importance.csv',index=False)
                pd.concat(predictions).to_csv(folder/'baseline_predictions.csv',index=False)
                pd.DataFrame(candidates).to_csv(folder/'svr_candidates.csv',index=False)
                U.save_json(dict(seed=seed,fold=fold,target=target,complete=True),folder/'complete.json')
                print('Controls complete',seed,fold,target,flush=True)
    for name,filename in [('fit_budget_svr_controls','fit_budget'),('point_baselines','point_baselines'),('feature_importance','feature_importance'),('svr_tuning_candidates','svr_candidates')]:
        frames=[pd.read_csv(p.parent/(filename+'.csv')) for p in checkpoint.glob('*/complete.json')]
        pd.concat(frames,ignore_index=True).to_csv(U.OUT/(name+'.csv'),index=False)
    (U.OUT/'predictions').mkdir(exist_ok=True)
    pd.concat([pd.read_csv(p.parent/'baseline_predictions.csv') for p in checkpoint.glob('*/complete.json')]).to_csv(U.OUT/'predictions/point_baselines.csv',index=False)
    print('All controls exported.',flush=True)

if __name__=='__main__':
    with threadpool_limits(limits=3):main()
