"""Validate complete experiment coverage and traceability of scientific outputs."""
import hashlib,json,itertools
import numpy as np
import pandas as pd
from revision_experiments import OUT,DATA,MODELS,SEEDS,TARGETS,ROOT

def main():
    folders=list((OUT/'runs').glob('*/complete.json'))
    records=[json.loads(p.read_text()) for p in folders]
    folds=['random_supplied','grouped_random','Re25','Re100','Re250','Re500']+[f'G0{i}' for i in range(1,7)]+['combined']
    expected=set(itertools.product(SEEDS,folds,TARGETS,MODELS))
    actual={(r['seed'],r['fold'],r['target'],r['model']) for r in records}
    assert actual==expected,('Missing:',expected-actual,'Unexpected:',actual-expected)
    assert len(records)==468
    predictions=0;assignment_hashes={}
    for p,meta in zip(folders,records):
        a=pd.read_csv(p.parent/'partitions.csv');d=pd.read_csv(p.parent/'predictions.csv');m=pd.read_csv(p.parent/'metrics.csv')
        assignment_hash=hashlib.sha256((p.parent/'partitions.csv').read_bytes()).hexdigest()
        assignment_key=(meta['seed'],meta['fold'])
        assert assignment_hashes.setdefault(assignment_key,assignment_hash)==assignment_hash
        if meta['fold']!='random_supplied':
            roles=['fit','tune','scale','cal','test']+(['id_test'] if meta['fold']!='grouped_random' else [])
            for i,part in enumerate(roles):
                for other in roles[:i]:
                    assert not set(a.loc[a.partition==part,'group_id'])&set(a.loc[a.partition==other,'group_id'])
        assert np.isfinite(d.prediction).all();assert np.isfinite(d.ape).all()
        # XGBoost prediction columns were serialized from float32; errors were
        # calculated before that decimal round-trip. Allow its rounding only.
        rounding=2e-5 if meta['model']=='XGBoost_seed' else 1e-9
        np.testing.assert_allclose(abs(d.prediction-d.truth)/np.maximum(abs(d.truth),1e-12)*100,d.ape,rtol=2e-6 if rounding>1e-9 else 1e-10,atol=rounding)
        assert len(d[d.role=='test'])==meta['counts']['test']
        assert (m.method=='random').any()
        assert m.coverage.between(0,1).all()
        assert m.loc[m.n_retained==0,'retained_mape'].isna().all()
        assert m.loc[m.n_retained==0,'retained_failure'].isna().all()
        for role,g in d.groupby('role'):
            row=m[(m.method=='UD_max')&(m.role==role)].iloc[0]
            np.testing.assert_allclose(g.accepted.mean(),row.coverage,atol=1e-12)
            np.testing.assert_allclose(g.ape.mean(),row.all_mape,atol=1e-10)
        predictions+=len(d)
    svr=pd.read_csv(OUT/'fit_budget_svr_controls.csv')
    assert (svr.loc[svr.model=='SVR_tuned','fit_status']==0).all()
    ci=pd.read_csv(OUT/'cluster_intervals.csv');assert len(ci)==156
    coverage=pd.read_csv(OUT/'conformal_group_coverage.csv');assert len(coverage)==144
    assert (coverage.group_coverage<=coverage.row_coverage+1e-12).all()
    report=dict(status='passed',experiments=len(records),prediction_rows=predictions,cluster_interval_rows=len(ci),conformal_coverage_records=len(coverage),
      source_sha256=hashlib.sha256((DATA/'02_processed_data/ML_dataset_longform.csv').read_bytes()).hexdigest(),
      code_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'scripts').glob('*revision*.py')})
    (OUT/'validation_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
