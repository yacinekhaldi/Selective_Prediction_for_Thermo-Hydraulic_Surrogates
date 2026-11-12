"""Independent GP normalization and deterministic CQR replay on U-bend."""
import numpy as np
import pandas as pd
from scipy.linalg import solve_triangular
from threadpoolctl import threadpool_limits
import ubend_experiments as U

def main():
    df=pd.read_csv(U.DATA/'ubend_longform.csv');df['group_id']=df.sample_id
    parts=U.partitions(df,42,'grouped_random');records=[]
    gp=U.Model('GP_ARD','DelP_Pa',42).fit(parts['fit'],parts['tune'])
    model=gp.members[0];x=gp.scaler.transform(U.xspace(parts['test']))
    m,s=model.predict(x,return_std=True);cross=model.kernel_(x,model.X_train_)
    manual_m=(cross@model.alpha_)*model._y_train_std+model._y_train_mean
    v=solve_triangular(model.L_,cross.T,lower=True)
    manual_s=np.sqrt(np.maximum(0,model.kernel_.diag(x)-(v*v).sum(axis=0)))*model._y_train_std
    np.testing.assert_allclose(m,manual_m,rtol=1e-10,atol=1e-10)
    np.testing.assert_allclose(s,manual_s,rtol=1e-7,atol=1e-9)
    stored=pd.read_csv(U.OUT/'runs/42_grouped_random_DelP_Pa_GP_ARD/predictions.csv').query("role == 'test'")
    np.testing.assert_allclose(gp.inverse(m),stored.prediction,rtol=1e-7,atol=1e-7)
    records.append(dict(check='GP unstandardized pressure mean/std and inverse-log median replay',seed=42,fold='grouped_random',target='DelP_Pa',n=len(stored),max_mean_difference=float(abs(m-manual_m).max()),max_std_difference=float(abs(s-manual_s).max()),median_guard_count=int(((m < -30)|(m > 30)).sum()),variance_guard_count=int(((s*s>30)|(2*m+s*s < -60)|(2*m+s*s>60)).sum())))
    del gp,model,cross,v
    for target in U.TARGETS:
        model=U.Model('CQR',target,42).fit(parts['fit'],parts['tune'])
        lo,hi=model.predict(parts['cal'])[2];y=parts['cal'][target].to_numpy()
        block=pd.DataFrame({'g':parts['cal'].group_id.to_numpy(),'r':np.maximum(lo-y,y-hi)}).groupby('g').r.max().to_numpy()
        rank=int(np.ceil(.9*(len(block)+1)));correction=max(0.,np.sort(block)[rank-1]) if rank<=len(block) else np.inf
        folder=U.OUT/'runs'/f'42_grouped_random_{target}_CQR'
        import json
        meta=json.loads((folder/'complete.json').read_text())
        np.testing.assert_allclose(correction,meta['conformal_correction'],rtol=1e-12,atol=1e-12)
        point,_,bounds=model.predict(parts['test']);stored=pd.read_csv(folder/'predictions.csv').query("role == 'test'")
        endpoints=pd.read_csv(folder/'interval_endpoints.csv').query("role == 'test'")
        np.testing.assert_allclose(point,stored.prediction,rtol=1e-12,atol=1e-12)
        np.testing.assert_allclose(bounds[0]-correction,endpoints.lower,rtol=1e-12,atol=1e-12)
        np.testing.assert_allclose(bounds[1]+correction,endpoints.upper,rtol=1e-12,atol=1e-12)
        records.append(dict(check='CQR group-quantile correction and endpoints replay',seed=42,fold='grouped_random',target=target,calibration_groups=len(block),rank=rank,correction=float(correction),point_and_endpoints_match=True))
    U.save_json(dict(status='passed',scope='seed-42 design-random GP pressure and CQR both targets; no assertion about all-fold Gaussian guard inactivity',records=records),U.OUT/'transform_audit.json')
    print('GP and CQR numerical replay audits passed.',flush=True)

if __name__=='__main__':
    with threadpool_limits(limits=3):main()
