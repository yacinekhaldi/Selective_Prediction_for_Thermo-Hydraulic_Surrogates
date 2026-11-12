"""Validate complete U-bend checkpoints, predictions, calibration and exports."""
import argparse,hashlib,itertools,json
from pathlib import Path
import numpy as np
import pandas as pd
import ubend_experiments as U

def main(require_exports=False):
    source=pd.read_csv(U.DATA/'ubend_longform.csv').set_index('case_uid')
    expected=set(itertools.product(U.SEEDS,U.FOLDS,U.TARGETS,U.MODELS))
    paths=sorted((U.OUT/'runs').glob('*/complete.json'))
    meta=[json.loads(p.read_text()) for p in paths]
    actual={(r['seed'],r['fold'],r['target'],r['model']) for r in meta}
    assert actual==expected,(len(actual),expected-actual)
    assignment_hashes={};planned={};prediction_rows=0;metric_rows=0
    design=source.reset_index();design['group_id']=design.sample_id
    for path,record in zip(paths,meta):
        folder=path.parent
        assignments=pd.read_csv(folder/'partitions.csv');pred=pd.read_csv(folder/'predictions.csv');m=pd.read_csv(folder/'metrics.csv');cal=pd.read_csv(folder/'calibration_scores.csv')
        key=(record['seed'],record['fold']);h=hashlib.sha256((folder/'partitions.csv').read_bytes()).hexdigest()
        assert assignment_hashes.setdefault(key,h)==h,(key,record['model'])
        roles=['fit','tune','scale','cal','test']+(['id_test'] if record['fold'] not in ['random_supplied','grouped_random'] else [])
        if key not in planned:
            planned[key]={role:set(d.case_uid) for role,d in U.partitions(design,*key).items()}
        for role in roles:
            assert set(assignments.loc[assignments.partition==role,'case_uid'])==planned[key][role],(key,role)
        fit_ids=assignments.loc[assignments.partition=='fit','case_uid']
        assert hashlib.sha256('\n'.join(fit_ids).encode()).hexdigest()==record['fit_case_hash']
        ids={role:set(assignments.loc[assignments.partition==role,'group_id']) for role in roles}
        assert all(not ids[a]&ids[b] for i,a in enumerate(roles) for b in roles[:i])
        assert set(cal.case_uid)==set(assignments.loc[assignments.partition=='cal','case_uid'])
        assert len(cal)==record['counts']['cal']
        assert np.isfinite(pred[['prediction','truth','ape']].to_numpy()).all()
        assert (pred.truth>0).all()
        np.testing.assert_allclose(pred.truth,source.loc[pred.case_uid,record['target']],rtol=1e-13,atol=1e-13)
        np.testing.assert_allclose(pred[U.FEATURES],source.loc[pred.case_uid,U.FEATURES],rtol=1e-13,atol=1e-13)
        rounding=2e-5 if record['model']=='XGBoost_seed' else 1e-9
        np.testing.assert_allclose(U.ape(pred.truth.to_numpy(),pred.prediction.to_numpy()),pred.ape,rtol=2e-6 if rounding>1e-9 else 1e-10,atol=rounding)
        tol=5. if record['target']=='CoolingObjective_K2m2' else 10.
        for role,g in pred.groupby('role'):
            assert len(g)==record['counts'][role]
            assert set(g.case_uid)==set(assignments.loc[assignments.partition==role,'case_uid'])
            row=m[(m.role==role)&(m.method=='UD_max')].iloc[0]
            np.testing.assert_allclose([g.accepted.mean(),g.ape.mean()],[row.coverage,row.all_mape],rtol=1e-11,atol=1e-10)
            assert g.case_uid.is_unique
            if g.accepted.any():
                np.testing.assert_allclose(g.loc[g.accepted,'ape'].mean(),row.retained_mape,rtol=1e-10,atol=1e-10)
                np.testing.assert_allclose((g.loc[g.accepted,'ape']>tol).mean(),row.retained_failure,rtol=1e-11,atol=1e-11)
            else:assert pd.isna(row.retained_mape) and pd.isna(row.retained_failure)
            accepted=g.UD_max<=row.threshold
            mismatched=accepted!=g.accepted
            assert np.isclose(g.loc[mismatched,'UD_max'],row.threshold,rtol=1e-12,atol=1e-12).all()
            for r in m[m.role==role].itertuples():
                if r.method in ['all','oracle']:continue
                cutoff=np.quantile(cal[r.method],.9,method='higher')
                np.testing.assert_allclose(cutoff,r.threshold,rtol=1e-12,atol=1e-12)
                assert r.n==len(g) and 0<=r.coverage<=1
                fail=g.ape.to_numpy()>tol
                if len(np.unique(fail))<2:assert pd.isna(r.auroc) and pd.isna(r.auprc)
                if np.ptp(g[r.method])==0:assert pd.isna(r.aurc)
        if record['model']=='CQR':
            ep=pd.read_csv(folder/'interval_endpoints.csv')
            truth=source.loc[ep.case_uid,record['target']].to_numpy()
            np.testing.assert_array_equal(ep.covered,(truth>=ep.lower)&(truth<=ep.upper))
            for role,z in ep.groupby('role'):
                values=m.loc[m.role==role,'interval_coverage'].dropna()
                np.testing.assert_allclose(values,z.covered.mean(),rtol=1e-12,atol=1e-12)
        prediction_rows+=len(pred);metric_rows+=len(m)
    aggregate=pd.read_csv(U.OUT/'metrics.csv');assert len(aggregate)==metric_rows
    ci=pd.read_csv(U.OUT/'cluster_intervals.csv');assert len(ci)==156
    coarse=pd.read_csv(U.OUT/'coarse_cluster_intervals.csv');assert len(coarse)==312
    cqr=pd.read_csv(U.OUT/'conformal_group_coverage.csv');assert len(cqr)==144
    np.testing.assert_allclose(cqr.row_coverage,cqr.group_coverage,rtol=1e-12,atol=1e-12)
    mixtures=pd.read_csv(U.OUT/'mixed_shift.csv');assert len(mixtures)==1980
    binary=pd.read_csv(U.OUT/'binary_coverage_sweep.csv');primary=aggregate[aggregate.method=='UD_max']
    check=binary[binary.calibration_quantile==.9].merge(primary,on=['seed','fold','target','model','role'],suffixes=('_binary','_main'),validate='one_to_one')
    assert len(check)==len(primary)
    for col in ['coverage','retained_mape','retained_failure','threshold']:np.testing.assert_allclose(check[col+'_binary'],check[col+'_main'],rtol=1e-10,atol=1e-10,equal_nan=True)
    budget=pd.read_csv(U.OUT/'fit_budget_svr_controls.csv');assert len(budget)==36
    svr=budget[budget.model=='SVR_tuned'];assert len(svr)==12 and (svr.fit_status==0).all()
    baselines=pd.read_csv(U.OUT/'point_baselines.csv');assert len(baselines)==60
    assert (baselines.fit_status==0).all()
    assert len(pd.read_csv(U.OUT/'svr_tuning_candidates.csv'))==108
    assert len(pd.read_csv(U.OUT/'feature_importance.csv'))==112
    export_info={}
    if require_exports:
        import joblib
        baseline_predictions=pd.read_csv(U.OUT/'predictions/point_baselines.csv')
        saved_models=sorted((U.OUT/'models').glob('*.joblib'));assert len(saved_models)==10
        for path in saved_models:
            artifact=joblib.load(path)
            assert artifact['features']==U.FEATURES
            name=path.stem.removesuffix('_'+artifact['target'])
            subset=baseline_predictions[(baseline_predictions.seed==42)&(baseline_predictions.fold=='random_supplied')&(baseline_predictions.target==artifact['target'])&(baseline_predictions.model==name)].head(12)
            assert len(subset)==12
            x=source.loc[subset.case_uid,U.FEATURES].to_numpy()
            values=artifact['estimator'].predict(x)
            if artifact['target']=='DelP_Pa':values=np.expm1(np.clip(values,-30,30))
            np.testing.assert_allclose(values,subset.prediction,rtol=1e-6,atol=1e-6)
            del artifact
        manifest=json.loads((U.OUT/'export_manifest.json').read_text())
        for name in manifest['tables']:
            for suffix in ['csv','tex']:assert (U.OUT/'tables'/(name+'.'+suffix)).is_file()
        for name in manifest['figures']:
            assert (U.OUT/'figures'/name).read_bytes().startswith(b'%PDF-')
            assert (U.OUT/'figures'/Path(name).with_suffix('.png')).is_file()
            assert (U.OUT/'plot_data'/Path(name).with_suffix('.csv')).is_file()
        pdf=json.loads((U.OUT/'reports/pdf_checks.json').read_text())
        assert all(not p['layout_warnings'] for p in pdf),pdf
        export_info=dict(tables=len(manifest['tables']),vector_figures=len(manifest['figures']),plot_data=len(manifest['plot_data']),saved_point_models_verified=len(saved_models))
    transform=json.loads((U.OUT/'transform_audit.json').read_text());assert transform['status']=='passed'
    report=dict(status='passed',experiments=len(meta),prediction_rows=prediction_rows,metric_rows=metric_rows,design_bootstrap_records=len(ci),coarse_bootstrap_records=len(coarse),conformal_coverage_records=len(cqr),mixed_configurations=len(mixtures),controls=len(budget),point_baselines=len(baselines),same_assignments_all_models_and_targets=True,all_roles_disjoint=True,calibration_cutoffs_verified=True,source_sha256=hashlib.sha256((U.DATA/'ubend_longform.csv').read_bytes()).hexdigest(),code_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (U.ROOT/'scripts').glob('*ubend*.py')},exports=export_info)
    U.save_json(report,U.OUT/'validation_report.json');print(json.dumps(report,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--with-exports',action='store_true');args=p.parse_args();main(args.with_exports)
