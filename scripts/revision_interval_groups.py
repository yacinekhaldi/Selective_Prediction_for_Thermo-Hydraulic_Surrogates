"""Export CQR interval endpoints and simultaneous group coverage."""
import json
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from revision_experiments import OUT,DATA,GROUP,Model,partitions

def main():
    df=pd.read_csv(DATA/'02_processed_data/ML_dataset_longform.csv');df['group_id']=df.groupby(GROUP).ngroup()
    rows=[]
    for p in sorted((OUT/'runs').glob('*_CQR/complete.json')):
        meta=json.loads(p.read_text());output=p.parent/'interval_endpoints.csv'
        if not output.exists():
            parts=partitions(df,meta['seed'],meta['fold']);model=Model('CQR',meta['target'],meta['seed']).fit(parts['fit'],parts['tune'])
            correction=meta['conformal_correction'];allrecords=[]
            old=pd.read_csv(p.parent/'predictions.csv')
            for role in old.role.unique():
                d=parts[role];pred,_,bounds=model.predict(d)
                # Check that the refit is the exact estimator underlying the metrics.
                np.testing.assert_allclose(pred,old.loc[old.role==role,'prediction'].to_numpy(),rtol=1e-12,atol=1e-12)
                lo,hi=bounds;truth=d[meta['target']].to_numpy()
                rec=d[['case_uid','group_id']].copy();rec['role']=role;rec['lower']=lo-correction;rec['upper']=hi+correction;rec['covered']=(truth>=rec.lower)&(truth<=rec.upper);allrecords.append(rec)
            pd.concat(allrecords).to_csv(output,index=False)
        data=pd.read_csv(output)
        for role,z in data.groupby('role'):
            base={k:meta[k] for k in ['seed','fold','target','model']}
            rows.append(dict(**base,role=role,rows=len(z),groups=z.group_id.nunique(),row_coverage=z.covered.mean(),group_coverage=z.groupby('group_id').covered.all().mean(),mean_width=(z.upper-z.lower).mean()))
    pd.DataFrame(rows).to_csv(OUT/'conformal_group_coverage.csv',index=False)
    print('CQR group coverage complete for',len(rows),'role records',flush=True)

if __name__=='__main__':
    with threadpool_limits(limits=2):main()
