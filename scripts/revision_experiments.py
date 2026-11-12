"""Resumable wavy-channel selective-prediction experiments.

Run with python scripts/revision_experiments.py. Checkpoints are resumable.
All tuning uses tune labels; normalization uses scale features; thresholds and
conformal corrections use calibration data. Outer test labels never select models.
"""
from __future__ import annotations
import argparse, hashlib, json, os, platform, sys, time, warnings
from pathlib import Path
os.environ.setdefault('OMP_NUM_THREADS', '4')
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVR
from threadpoolctl import threadpool_limits
from xgboost import XGBRegressor

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'data/raw/mendeley_wavy_channel/ML-CFD-Wavy-Channel-Surrogate'
OUT=ROOT/'results/revision_round1'
FEATURES=['Re','Pr','Da','epsi','Hp_mm','a_mm','Lw_mm']
GROUP=[f for f in FEATURES if f!='Pr']
TARGETS=['Nuavg','DelP_Pa']
MODELS=['ExtraTrees_seed','ExtraTrees_bootstrap','XGBoost_seed','GP_ARD','LogPolynomial','CQR']
SEEDS=[42,137,271]

def save_json(x,p):
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(x,indent=2,default=lambda x:x.item() if hasattr(x,'item') else str(x)),encoding='utf-8')

def xspace(d):
    x=d[FEATURES].to_numpy(float).copy(); x[:,:3]=np.log10(x[:,:3]); return x

def ape(y,p): return np.abs(p-y)/np.maximum(np.abs(y),1e-12)*100

def tie_curve(score,errors):
    """Expected prefix risk under uniform random ordering inside each score tie.

    Each member of a tie receives that tie group's mean error before cumulation.
    The discrete mean of prefix risks covers k=1,...,n. It is row-order invariant.
    """
    score=np.asarray(score); errors=np.asarray(errors)
    levels,inv,counts=np.unique(score,return_inverse=True,return_counts=True)
    sums=np.bincount(inv,weights=errors)
    ordered=np.repeat(sums/counts,counts)
    return np.arange(1,len(score)+1)/len(score),np.cumsum(ordered)/np.arange(1,len(score)+1)

def metrics(score,err,accepted,tol):
    fail=err>tol; n=len(err); k=int(accepted.sum()); classes=len(np.unique(fail))
    c,r=tie_curve(score,err)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        rho=float(spearmanr(score,err).statistic) if np.ptp(score)>0 and np.ptp(err)>0 else np.nan
    return dict(n=n,n_retained=k,coverage=k/n,all_mape=float(err.mean()),
        retained_mape=float(err[accepted].mean()) if k else np.nan,
        retained_failure=float(fail[accepted].mean()) if k else np.nan,
        false_accept=float((fail&accepted).mean()),false_abstain=float((~fail&~accepted).mean()),
        failure_prevalence=float(fail.mean()),auroc=float(roc_auc_score(fail,score)) if classes==2 else np.nan,
        auprc=float(average_precision_score(fail,score)) if classes==2 else np.nan,
        metric_status='single_failure_class' if classes<2 else ('constant_score' if np.ptp(score)==0 else 'defined'),
        spearman=rho,aurc=float(r.mean()) if np.ptp(score)>0 else np.nan)

def partitions(df,seed,fold):
    rng=np.random.default_rng(seed)
    if fold=='random_supplied':
        s=pd.read_csv(DATA/'03_splits/random80_20_split.csv').set_index('case_uid')['split']
        outer=df[s.reindex(df.case_uid).to_numpy()=='train'].copy()
        test=df[s.reindex(df.case_uid).to_numpy()=='test'].copy()
        ids=rng.permutation(outer.index)
        cuts=np.cumsum([.60,.10,.15])*len(ids)
        chunks=np.split(ids,cuts.astype(int))
        return {**dict(zip(['fit','tune','scale','cal'],[df.loc[a].copy() for a in chunks])), 'test':test,'id_test':test.iloc[:0]}
    if fold=='grouped_random': pool=df; ood=df.iloc[:0]
    elif fold.startswith('Re'):
        v=int(fold[2:]); pool=df[df.Re!=v]; ood=df[df.Re==v]
    elif fold.startswith('G'):
        pool=df[df.geometry_id!=fold]; ood=df[df.geometry_id==fold]
    else:
        pool=df[(df.geometry_id!='G06')&(df.Re!=500)]; ood=df[(df.geometry_id=='G06')&(df.Re==500)]
    groups=rng.permutation(pool.group_id.unique()); n=len(groups)
    chunks=np.split(groups,(np.array([.48,.56,.68,.80])*n).astype(int))
    parts={name:pool[pool.group_id.isin(g)].copy() for name,g in zip(['fit','tune','scale','cal','id_test'],chunks)}
    parts['test']=parts['id_test'].copy() if fold=='grouped_random' else ood.copy()
    sets=[set(parts[k].group_id) for k in ['fit','tune','scale','cal','test']]
    assert all(not (sets[i]&sets[j]) for i in range(5) for j in range(i)),fold
    return parts

class Model:
    def __init__(self,name,target,seed): self.name=name; self.target=target; self.seed=seed; self.members=[]; self.meta={}
    def inverse(self,z): return np.expm1(np.clip(z,-30,30)) if self.target=='DelP_Pa' else z
    def fit(self,fit,tune):
        start=time.perf_counter(); self.fitdf=fit
        self.scaler=StandardScaler().fit(xspace(fit))
        x=self.scaler.transform(xspace(fit)); xt=self.scaler.transform(xspace(tune))
        y=fit[self.target].to_numpy(); yt=tune[self.target].to_numpy()
        z=np.log1p(y) if self.target=='DelP_Pa' else y
        rng=np.random.default_rng(self.seed)
        if 'ExtraTrees' in self.name or 'XGBoost' in self.name:
            for j in range(5):
                seed=self.seed+17*j
                if 'ExtraTrees' in self.name:
                    m=ExtraTreesRegressor(n_estimators=200,min_samples_leaf=2,n_jobs=4,random_state=seed)
                else:
                    m=XGBRegressor(n_estimators=450,max_depth=6,learning_rate=.03,subsample=.95,colsample_bytree=.95,n_jobs=4,tree_method='hist',random_state=seed)
                ix=rng.integers(0,len(x),len(x)) if 'bootstrap' in self.name else np.arange(len(x))
                m.fit(x[ix],z[ix]); self.members.append(m)
            preds=self.member_predictions(tune)
            cor=np.corrcoef(preds.T); self.meta['mean_member_correlation']=float(cor[np.triu_indices(5,1)].mean())
            self.meta['tune_mean_relative_std']=float((preds.std(1)/np.maximum(abs(preds.mean(1)),1e-9)).mean())
        elif self.name=='GP_ARD':
            # Full fit set; tune two ARD initializations, then optimize likelihood
            # on fit labels only. A bounded optimizer makes cost reproducible.
            def optimizer(obj,theta,bounds):
                res=minimize(obj,theta,jac=True,bounds=bounds,method='L-BFGS-B',options={'maxiter':20,'ftol':1e-6})
                return res.x,res.fun
            best=None
            hyper_idx=np.sort(rng.choice(len(x),min(512,len(x)),replace=False))
            for ls in [1.,3.]:
                kernel=ConstantKernel(1.,(.01,100.))*RBF(np.full(7,ls),(.15,30.))
                m=GaussianProcessRegressor(kernel=kernel,alpha=1e-6,normalize_y=True,optimizer=optimizer,random_state=self.seed)
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore'); m.fit(x[hyper_idx],z[hyper_idx])
                loss=ape(yt,self.inverse(m.predict(xt))).mean()
                if best is None or loss<best[0]:best=(loss,m)
            final=GaussianProcessRegressor(kernel=best[1].kernel_,alpha=1e-6,normalize_y=True,optimizer=None)
            final.fit(x,z)
            self.members=[final]; self.meta['kernel']=str(final.kernel_)
            self.meta['hyperparameter_subset']=len(hyper_idx)
        elif self.name=='LogPolynomial':
            best=None
            for degree in [2,3]:
                for alpha in [.001,.1,10.]:
                    m=make_pipeline(PolynomialFeatures(degree,include_bias=False),Ridge(alpha=alpha))
                    m.fit(x,np.log(y))
                    loss=ape(yt,np.exp(np.clip(m.predict(xt),-30,30))).mean()
                    if best is None or loss<best[0]:best=(loss,m,degree,alpha)
            self.members=[best[1]]; self.meta.update(degree=best[2],ridge_alpha=best[3])
        else:
            for q in [.05,.5,.95]:
                m=HistGradientBoostingRegressor(loss='quantile',quantile=q,max_iter=250,max_leaf_nodes=15,l2_regularization=1.,random_state=self.seed)
                m.fit(x,z); self.members.append(m)
        self.meta['fit_seconds']=time.perf_counter()-start
        return self
    def member_predictions(self,d):
        x=self.scaler.transform(xspace(d))
        return np.column_stack([self.inverse(m.predict(x)) for m in self.members])
    def predict(self,d):
        x=self.scaler.transform(xspace(d))
        if self.name=='GP_ARD':
            z,s=self.members[0].predict(x,return_std=True)
            if self.target=='DelP_Pa':
                # Exact median and lognormal standard deviation after log1p fit.
                p=self.inverse(z); std=np.sqrt(np.expm1(np.minimum(s*s,30))*np.exp(np.clip(2*z+s*s,-60,60)))
            else:p=z; std=s
            return p,std,None
        if self.name=='LogPolynomial':
            p=np.exp(np.clip(self.members[0].predict(x),-30,30)); return p,np.zeros(len(d)),None
        p=self.member_predictions(d)
        if self.name=='CQR':
            lo=np.minimum(p[:,0],p[:,2]); hi=np.maximum(p[:,0],p[:,2]); return p[:,1],(hi-lo)/2,(lo,hi)
        return p.mean(1),p.std(1),None

class Support:
    def __init__(self,fit):
        self.scaler=StandardScaler().fit(xspace(fit)); self.x=self.scaler.transform(xspace(fit))
        self.nn=NearestNeighbors(n_neighbors=5).fit(self.x); self.cov=LedoitWolf().fit(self.x)
        z=np.column_stack([np.ones(len(self.x)),self.x]); self.inv=np.linalg.pinv(z.T@z)
        self.levels=np.unique(np.log10(fit.Re))
    def score(self,d):
        x=self.scaler.transform(xspace(d)); dist,ix=self.nn.kneighbors(x)
        z=np.column_stack([np.ones(len(x)),x]); lev=np.einsum('ij,jk,ik->i',z,self.inv,z)
        # Gap relative to nearest observed Re spacing, no test outputs used.
        r=np.log10(d.Re.to_numpy()); levels=self.levels
        gap=np.min(abs(r[:,None]-levels),axis=1)
        spacing=np.median(np.diff(levels)) if len(levels)>1 else 1.
        return dict(knn=dist.mean(1),maha=self.cov.mahalanobis(x),leverage=lev,support_gap=gap/spacing)

def scaled(a,reference):
    low,high=np.quantile(reference,[.05,.95]); span=high-low
    if span<1e-12:span=1.
    return (np.asarray(a)-low)/span

def physical(model,d,fit):
    p,_,_=model.predict(d); y=fit[model.target].to_numpy(); span=max(np.ptp(y),1e-9)
    v=(np.maximum(0,y.min()-.05*span-p)+np.maximum(0,p-y.max()-.05*span)+np.maximum(0,-p))/span
    # Counterfactual predictions at training Re levels only.
    prev=None; mono=np.zeros(len(d))
    for re in sorted(fit.Re.unique()):
        counter=d.copy(); counter['Re']=re; pred=model.predict(counter)[0]
        if prev is not None:mono+=np.maximum(0,prev-pred)/span
        prev=pred
    return v+mono

def score_frames(model,parts):
    support=Support(parts['fit']); raw={}; predictions={}; ranges={}
    for key in ['tune','scale','cal','test','id_test']:
        d=parts[key]
        if len(d)==0:continue
        p,s,bounds=model.predict(d); predictions[key]=p; ranges[key]=bounds
        raw[key]=support.score(d); raw[key]['U']=s/np.maximum(abs(p),1e-9)
        raw[key]['V']=physical(model,d,parts['fit'])
    scores={}
    rng=np.random.default_rng(model.seed+9221)
    for key,r in raw.items():
        u=scaled(r['U'],raw['scale']['U']); dk=scaled(r['knn'],raw['scale']['knn']); dm=scaled(r['maha'],raw['scale']['maha'])
        d=np.maximum(dk,dm); v=scaled(r['V'],raw['scale']['V'])
        scores[key]={'U':u,'D':d,'V':v,'UD_max':np.maximum(u,d),'UD_equal':(u+d)/2,
          'UDV_max':np.maximum.reduce([u,d,v]),'UDV_equal':(u+d+v)/3,
          'leverage':r['leverage'],'support_gap':r['support_gap'],
          'legacy_clipped':np.maximum.reduce([np.clip(u,0,1),np.clip(d,0,1),np.clip(v,0,1)]),
          'random':rng.random(len(u))}
    tolerance=5. if model.target=='Nuavg' else 10.
    f=ape(parts['tune'][model.target].to_numpy(),predictions['tune'])>tolerance
    if len(np.unique(f))==2:
        gate=make_pipeline(StandardScaler(),LogisticRegression(C=1.,max_iter=1000,random_state=model.seed))
        gate.fit(np.column_stack([scores['tune'][n] for n in ['U','D','V']]),f)
        for k in scores:scores[k]['learned']=gate.predict_proba(np.column_stack([scores[k][n] for n in ['U','D','V']]))[:,1]
    if model.name=='CQR':
        lo,hi=ranges['cal']; y=parts['cal'][model.target].to_numpy()
        residual=np.maximum(lo-y,y-hi)
        # Max residual within each non-Pr group protects all Pr variants.
        block=pd.DataFrame({'group':parts['cal'].group_id.to_numpy(),'r':residual}).groupby('group').r.max().to_numpy()
        rank=int(np.ceil((len(block)+1)*.90))
        correction=float(np.sort(block)[rank-1]) if rank<=len(block) else np.inf
        correction=max(0.,correction)
        model.meta['conformal_calibration_groups']=len(block)
        model.meta['conformal_correction']=correction
        for k in scores:
            lo,hi=ranges[k]; lo=lo-correction; hi=hi+correction
            ranges[k]=(lo,hi); scores[k]['CQR_width']=(hi-lo)/np.maximum(abs(predictions[k]),1e-9)
    return scores,predictions,raw,ranges

def one_experiment(df,seed,fold,target,name):
    folder=OUT/'runs'/f'{seed}_{fold}_{target}_{name}'
    if (folder/'complete.json').exists():return
    folder.mkdir(parents=True,exist_ok=True)
    parts=partitions(df,seed,fold)
    assignment=pd.concat([d[['case_uid','group_id']].assign(partition=k) for k,d in parts.items() if k!='id_test' or fold!='grouped_random'])
    assignment.to_csv(folder/'partitions.csv',index=False)
    model=Model(name,target,seed).fit(parts['fit'],parts['tune'])
    scores,preds,raw,ranges=score_frames(model,parts)
    # Same-fit single member versus ensemble controls the training-size difference.
    if 'seed' in name or 'bootstrap' in name:
        model.meta['test_single_member_mape']=float(ape(parts['test'][target].to_numpy(),model.member_predictions(parts['test'])[:,0]).mean())
    rows=[]; sensitivity=[]; curves=[]; records=[]; matched=[]
    tol=5. if target=='Nuavg' else 10.
    for role in ['test','id_test']:
        if role not in scores or (fold=='grouped_random' and role=='id_test'):continue
        d=parts[role]; err=ape(d[target].to_numpy(),preds[role])
        rec=d[['case_uid','group_id','geometry_id']+FEATURES].copy(); rec['role']=role
        rec['truth']=d[target].to_numpy(); rec['prediction']=preds[role]; rec['ape']=err
        rec['raw_U']=raw[role]['U']; rec['raw_V']=raw[role]['V']; rec['raw_knn']=raw[role]['knn']; rec['raw_maha']=raw[role]['maha']
        rec['signed_error_pct']=100*(preds[role]-d[target].to_numpy())/d[target].to_numpy()
        for method,score in scores[role].items():
            calibration=scores['cal'][method]
            threshold=float(np.quantile(calibration,.90,method='higher'))
            accepted=score<=threshold
            row=dict(seed=seed,fold=fold,target=target,model=name,role=role,method=method,threshold=threshold,**metrics(score,err,accepted,tol))
            if name=='CQR':
                lo,hi=ranges[role]; row['interval_coverage']=float(((d[target].to_numpy()>=lo)&(d[target].to_numpy()<=hi)).mean())
                row['interval_width']=float((hi-lo).mean())
            rows.append(row);rec[method]=score
            if method=='UD_max':rec['accepted']=accepted;rec['threshold']=threshold
            c,r=tie_curve(score,err)
            for coverage in [.5,.7,.8,.9,.95]:
                ix=min(len(c)-1,int(np.ceil(coverage*len(c)))-1)
                matched.append(dict(seed=seed,fold=fold,target=target,model=name,role=role,method=method,coverage=c[ix],mape=r[ix]))
            if seed==42 and method in ['U','D','UD_max','UD_equal','legacy_clipped','CQR_width']:
                for ix in np.unique(np.linspace(0,len(c)-1,101).astype(int)):
                    curves.append(dict(fold=fold,target=target,model=name,role=role,method=method,coverage=c[ix],risk=r[ix]))
        for q1 in [.60,.75,.80,.85]:
            for q2 in [.85,.90,.95,.98]:
                if q1>=q2:continue
                sp=scores[role]['UD_max']; cal=scores['cal']['UD_max']
                t1,t2=np.quantile(cal,[q1,q2],method='higher'); state=np.where(sp<=t1,'predict',np.where(sp<=t2,'warn','abstain'))
                for st in ['predict','warn','abstain']:
                    keep=state==st
                    sensitivity.append(dict(seed=seed,fold=fold,target=target,model=name,role=role,q_predict=q1,q_warn=q2,tau_predict=t1,tau_warn=t2,state=st,n=int(keep.sum()),coverage=float(keep.mean()),mape=float(err[keep].mean()) if keep.any() else np.nan,failure=float((err[keep]>tol).mean()) if keep.any() else np.nan))
        # All-case and oracle are explicit retrospective references.
        rows.append(dict(seed=seed,fold=fold,target=target,model=name,role=role,method='all',threshold=np.nan,**metrics(np.zeros(len(err)),err,np.ones(len(err),bool),tol)))
        rows.append(dict(seed=seed,fold=fold,target=target,model=name,role=role,method='oracle',threshold=np.nan,**metrics(err,err,err<=np.quantile(err,.9),tol)))
        records.append(rec)
    pd.DataFrame(rows).to_csv(folder/'metrics.csv',index=False)
    pd.DataFrame(sensitivity).to_csv(folder/'sensitivity.csv',index=False)
    pd.DataFrame(curves).to_csv(folder/'curves.csv',index=False)
    pd.DataFrame(matched).to_csv(folder/'matched.csv',index=False)
    pd.concat(records).to_csv(folder/'predictions.csv',index=False)
    model.meta.update(seed=seed,fold=fold,target=target,model=name,counts={k:len(v) for k,v in parts.items()},fit_case_hash=hashlib.sha256('\n'.join(parts['fit'].case_uid).encode()).hexdigest())
    save_json(model.meta,folder/'complete.json')
    print(f'DONE {seed} {fold} {target} {name} fit={model.meta["fit_seconds"]:.1f}s',flush=True)

def audit(df):
    OUT.mkdir(parents=True,exist_ok=True)
    grouped=df.groupby(GROUP).DelP_Pa.agg(['min','max','nunique'])
    grouped['relative_range_pct']=100*(grouped['max']-grouped['min'])/grouped['max']
    grouped.to_csv(OUT/'pressure_pr_audit.csv')
    df[['geometry_id','a_mm','Lw_mm']].drop_duplicates().to_csv(OUT/'geometry_mapping.csv',index=False)
    split=pd.read_csv(DATA/'03_splits/random80_20_split.csv').set_index('case_uid')['split']
    train=df[split.reindex(df.case_uid).to_numpy()=='train']; test=df[split.reindex(df.case_uid).to_numpy()=='test']
    matching=test.group_id.isin(train.group_id)
    duplicates=sum(any(np.isclose(r.DelP_Pa,train.loc[train.group_id==r.group_id,'DelP_Pa'],rtol=1e-12,atol=1e-12)) for r in test.itertuples())
    save_json(dict(rows=len(df),groups=len(grouped),constant_pressure_groups=int((grouped['nunique']==1).sum()),test_rows=len(test),overlap_non_pr_rows=int(matching.sum()),matching_pressure_rows=duplicates,source_sha256=hashlib.sha256((DATA/'02_processed_data/ML_dataset_longform.csv').read_bytes()).hexdigest()),OUT/'audit.json')

def aggregate():
    for file in ['metrics','sensitivity','curves','matched']:
        frames=[]
        for p in sorted((OUT/'runs').glob(f'*/{file}.csv')):
            if (p.parent/'complete.json').exists() and p.stat().st_size>5:
                try:frames.append(pd.read_csv(p))
                except pd.errors.EmptyDataError:pass
        if frames:pd.concat(frames,ignore_index=True).to_csv(OUT/f'{file}.csv',index=False)
    meta=[json.loads(p.read_text()) for p in (OUT/'runs').glob('*/complete.json')]
    save_json(meta,OUT/'run_manifest.json')
    import sklearn,scipy,xgboost
    save_json(dict(python=sys.version,platform=platform.platform(),numpy=np.__version__,pandas=pd.__version__,scipy=scipy.__version__,sklearn=sklearn.__version__,xgboost=xgboost.__version__,seeds=SEEDS,models=MODELS,completed_runs=len(meta)),OUT/'environment.json')

def main():
    p=argparse.ArgumentParser();p.add_argument('--seeds',nargs='+',type=int,default=SEEDS);p.add_argument('--models',nargs='+',default=MODELS);p.add_argument('--folds',nargs='+',default=['random_supplied','grouped_random','Re25','Re100','Re250','Re500']+[f'G0{i}' for i in range(1,7)]+['combined']);args=p.parse_args()
    df=pd.read_csv(DATA/'02_processed_data/ML_dataset_longform.csv');df['group_id']=df.groupby(GROUP,sort=True).ngroup()
    audit(df)
    with threadpool_limits(limits=4):
        for seed in args.seeds:
            for fold in args.folds:
                for target in TARGETS:
                    for model in args.models:one_experiment(df,seed,fold,target,model)
    aggregate()

if __name__=='__main__':main()
