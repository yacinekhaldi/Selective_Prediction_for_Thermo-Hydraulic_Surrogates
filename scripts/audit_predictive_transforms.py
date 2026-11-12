"""Replay existing GP/CQR settings to audit transformations and calibration."""
import json
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
import pandas as pd
from scipy.linalg import solve_triangular
from threadpoolctl import threadpool_limits
import revision_experiments as W
import ubend_experiments as B

def replay(job):
    system,seed,fold,target,name=job;u=W if system=='wavy' else B
    dest=u.OUT/'predictive_audit';dest.mkdir(exist_ok=True)
    record_path=dest/f'{seed}_{fold}_{target}_{name}.json'
    if record_path.exists():return json.loads(record_path.read_text())
    with threadpool_limits(limits=4 if system=='wavy' else 3):
        df=pd.read_csv(u.DATA/('02_processed_data/ML_dataset_longform.csv' if system=='wavy' else 'ubend_longform.csv'))
        df['group_id']=df.groupby(u.GROUP).ngroup() if system=='wavy' else df.sample_id
        parts=u.partitions(df,seed,fold);model=u.Model(name,target,seed).fit(parts['fit'],parts['tune'])
        folder=u.OUT/'runs'/f'{seed}_{fold}_{target}_{name}'
        meta=json.loads((folder/'complete.json').read_text());stored=pd.read_csv(folder/'predictions.csv')
        result=dict(system=system,seed=seed,fold=fold,target=target,model=name,status='passed')
        if name=='GP_ARD':
            counts=dict(median_guard=0,variance_s2_guard=0,variance_exponent_guard=0);mean_error=0.;std_error=0.;prediction_error=0.
            for role in ['test','id_test']:
                old=stored[stored.role==role]
                if old.empty:continue
                d=parts[role];x=model.scaler.transform(u.xspace(d));gp=model.members[0]
                m,s=gp.predict(x,return_std=True);cross=gp.kernel_(x,gp.X_train_)
                manual=(cross@gp.alpha_)*gp._y_train_std+gp._y_train_mean
                v=solve_triangular(gp.L_,cross.T,lower=True)
                manual_s=np.sqrt(np.maximum(0,gp.kernel_.diag(x)-(v*v).sum(axis=0)))*gp._y_train_std
                np.testing.assert_allclose(m,manual,rtol=1e-9,atol=1e-9)
                np.testing.assert_allclose(s,manual_s,rtol=1e-6,atol=1e-8)
                pred,std,_=model.predict(d)
                np.testing.assert_allclose(pred,old.prediction,rtol=1e-6,atol=1e-7)
                expected_std=old.raw_U.to_numpy()*np.maximum(abs(old.prediction.to_numpy()),1e-9)
                np.testing.assert_allclose(std,expected_std,rtol=1e-5,atol=1e-7)
                mean_error=max(mean_error,float(abs(m-manual).max()));std_error=max(std_error,float(abs(s-manual_s).max()))
                prediction_error=max(prediction_error,float(np.max(abs(pred-old.prediction.to_numpy())/np.maximum(abs(old.prediction.to_numpy()),1e-9))))
                counts['median_guard']+=int(((m < -30)|(m > 30)).sum())
                counts['variance_s2_guard']+=int((s*s>30).sum())
                counts['variance_exponent_guard']+=int(((2*m+s*s < -60)|(2*m+s*s >60)).sum())
            result.update(**counts,max_mean_difference=mean_error,max_std_difference=std_error,max_relative_prediction_replay_difference=prediction_error)
        else:
            lo,hi=model.predict(parts['cal'])[2];y=parts['cal'][target].to_numpy()
            scores=pd.DataFrame({'g':parts['cal'].group_id.to_numpy(),'r':np.maximum(lo-y,y-hi)}).groupby('g').r.max().to_numpy()
            n=len(scores);k=int(np.ceil(.9*(n+1)));q=max(0.,np.sort(scores)[k-1]) if k<=n else np.inf
            np.testing.assert_allclose(q,meta['conformal_correction'],rtol=1e-10,atol=1e-10)
            ep=pd.read_csv(folder/'interval_endpoints.csv');coverage=[]
            for role,z in ep.groupby('role'):
                d=parts[role];p,_,(lo,hi)=model.predict(d)
                np.testing.assert_allclose(lo-q,z.lower,rtol=1e-10,atol=1e-10)
                np.testing.assert_allclose(hi+q,z.upper,rtol=1e-10,atol=1e-10)
                covered=(d[target].to_numpy()>=z.lower.to_numpy())&(d[target].to_numpy()<=z.upper.to_numpy())
                np.testing.assert_array_equal(covered,z.covered)
                coverage.append(dict(role=role,row_coverage=float(covered.mean()),group_coverage=float(pd.DataFrame({'g':d.group_id.to_numpy(),'v':covered}).groupby('g').v.all().mean())))
            other=set().union(*(set(parts[role].group_id) for role in ['fit','tune','scale']))
            overlap=len(other&set(parts['cal'].group_id))
            if system=='ubend' or fold!='random_supplied':assert overlap==0
            result.update(calibration_groups=n,rank=k,correction=float(q),row_random_descriptive_only=fold=='random_supplied',fit_tune_scale_vs_cal_group_overlap=overlap,coverage=coverage)
        record_path.write_text(json.dumps(result,indent=2),encoding='utf-8');return result

def main():
    jobs=[]
    for system,u in [('wavy',W),('ubend',B)]:
        for p in (u.OUT/'runs').glob('*/complete.json'):
            m=json.loads(p.read_text())
            if m['model']=='CQR' or (m['model']=='GP_ARD' and m['target']=='DelP_Pa'):jobs.append((system,m['seed'],m['fold'],m['target'],m['model']))
    records=[]
    with ProcessPoolExecutor(max_workers=3) as pool:
        for future in as_completed([pool.submit(replay,j) for j in jobs]):
            records.append(future.result())
            print(f'Audited {len(records)}/{len(jobs)}',flush=True)
    for system,u in [('wavy',W),('ubend',B)]:
        records_for=[r for r in records if r['system']==system]
        gp=[r for r in records_for if r['model']=='GP_ARD'];cqr=[r for r in records_for if r['model']=='CQR']
        assert len(gp)==39 and len(cqr)==78
        pd.DataFrame(gp).to_csv(u.OUT/'gp_pressure_audit.csv',index=False)
        pd.DataFrame([{k:v for k,v in r.items() if k!='coverage'} for r in cqr]).to_csv(u.OUT/'cqr_calibration_audit.csv',index=False)
        summary=dict(status='passed',gp_pressure_runs=39,cqr_runs=78,guard_counts={k:sum(r[k] for r in gp) for k in ['median_guard','variance_s2_guard','variance_exponent_guard']},results_unchanged=True)
        (u.OUT/'predictive_audit_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
        print(system,summary,flush=True)

if __name__=='__main__':main()
