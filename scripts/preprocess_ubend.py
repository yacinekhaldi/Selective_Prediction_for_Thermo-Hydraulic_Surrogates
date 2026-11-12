"""Prepare physical-unit U-bend CSVs in the existing long-form dataset style."""
from __future__ import annotations
import hashlib,json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
RAW=ROOT/'data/raw/ubend_cfd'
OUT=ROOT/'data/preprocessed/ubend_cfd'
FEATURES=[f'input_{i}' for i in range(28)]
TARGET_MAP={'target_0':'DelP_Pa','target_1':'CoolingObjective_K2m2'}
METADATA=['case_uid','raw_row_id','sample_id','geometry_id']

def sha256(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def identifiers(frame):
    frame=frame.copy()
    frame['sample_id']=frame.sample_id.astype('int64')
    frame['case_uid']=frame.sample_id.map(lambda i:f'UBEND_{i:05d}')
    frame['raw_row_id']=frame.sample_id+1
    frame['geometry_id']=frame.sample_id.map(lambda i:f'U{i:05d}')
    return frame

def main():
    manifest=json.loads((RAW/'download_manifest.json').read_text(encoding='utf-8'))
    for record in manifest['files']:
        assert sha256(RAW/record['path'])==record['sha256'],record['path']
    frames={name:pd.read_parquet(RAW/'params'/f'{name}.parquet') for name in ['data','train','test','failed']}
    for name in ['train','test','failed']:
        z=frames[name]
        assert z.sample_id.is_unique
        reference=frames['data'].iloc[z.sample_id.to_numpy()].reset_index(drop=True)
        pd.testing.assert_frame_equal(reference,z[reference.columns].reset_index(drop=True))
    sets={n:set(frames[n].sample_id) for n in ['train','test','failed']}
    assert not sets['train']&sets['test'] and not sets['train']&sets['failed'] and not sets['test']&sets['failed']
    assert set.union(*sets.values())==set(range(len(frames['data'])))
    success=pd.concat([frames['train'].assign(source_split='train'),frames['test'].assign(source_split='test')],ignore_index=True)
    success=identifiers(success.sort_values('sample_id').reset_index(drop=True))
    numeric=FEATURES+list(TARGET_MAP)
    for col in numeric:success[col]=pd.to_numeric(success[col],errors='raise')
    assert np.isfinite(success[numeric].to_numpy()).all()
    assert success[list(TARGET_MAP)].gt(0).all().all()
    assert (success.target_0<10000).all()
    assert success.duplicated(FEATURES).sum()==0
    model_data=success[METADATA+numeric].rename(columns=TARGET_MAP)
    assignments=success[['case_uid','source_split']].rename(columns={'source_split':'split'})
    failed=identifiers(frames['failed'].sort_values('sample_id').reset_index(drop=True))
    assert failed.target_0.isin([10000,11000]).all()
    failed['failure_reason']=failed.target_0.map({10000:'mesh_generation_failed',11000:'solver_diverged'})
    failed=failed[METADATA+FEATURES+[c for c in frames['failed'].columns if c not in FEATURES+['sample_id']]+['failure_reason']]
    diagnostics=success[METADATA+['source_split','target_0_mean','target_0_std','target_1_mean','target_1_std']]
    OUT.mkdir(parents=True,exist_ok=True)
    outputs={
      'ubend_longform.csv':model_data,
      'random90_10_split.csv':assignments,
      'failed_simulations.csv':failed,
      'solver_diagnostics.csv':diagnostics,
      'numeric_summary.csv':model_data[FEATURES+list(TARGET_MAP.values())].describe().T.rename_axis('variable').reset_index()}
    for name,df in outputs.items():df.to_csv(OUT/name,index=False)
    # Validate the serialized files, not only the in-memory dataframes.
    loaded=pd.read_csv(OUT/'ubend_longform.csv')
    np.testing.assert_allclose(loaded[FEATURES+list(TARGET_MAP.values())],model_data[FEATURES+list(TARGET_MAP.values())],rtol=1e-13,atol=1e-14)
    assert loaded.case_uid.is_unique and len(loaded)==len(assignments)
    assert set(loaded.case_uid)==set(assignments.case_uid)
    config={
      'status':'passed','dataset':'U-bend CFD, updated Hugging Face params release',
      'source_repository':manifest['repository'],'source_revision':manifest['revision'],
      'source_file_sha256':{r['path']:r['sha256'] for r in manifest['files']},
      'rows':{'raw_total':len(frames['data']),'successful':len(model_data),'train':int((assignments.split=='train').sum()),'test':int((assignments.split=='test').sum()),'failed':len(failed)},
      'failure_counts':{str(k):int(v) for k,v in failed.failure_reason.value_counts().items()},
      'features':FEATURES,'targets':list(TARGET_MAP.values()),'target_mapping':TARGET_MAP,
      'target_units':{'DelP_Pa':'Pa','CoolingObjective_K2m2':'K^2 m^2 (integrated squared wall/inlet temperature difference)'},
      'metadata_columns':METADATA,'model_data':'ubend_longform.csv','supplied_split':'random90_10_split.csv',
      'preprocessing':{'numeric_conversion':'strict','nonfinite_model_values':0,'missing_model_values':0,'duplicate_input_rows':0,'imputation':'none','outlier_removal':'none; retain every successful simulation','input_transform':'identity: signed geometry parameters remain in their released coordinates','scaling':'none during file preprocessing; fit StandardScaler on fit rows only in model experiments','target_transform':'physical-unit labels retained; pressure log1p/inverse expm1 may be applied inside the model pipeline as in the original study','failed_runs':'separate; sentinel codes never treated as physical regression targets','solver_statistics':'separate sidecar; never input features','geometry_id':'unique design identifier, not a geometry-family classification','raw_row_id':'one-based aggregate row; sample_id is zero-based upstream row identifier'},
      'validation':{'source_hashes_verified':True,'source_split_disjoint':True,'split_union_matches_raw':True,'split_values_match_raw_by_sample_id':True,'csv_roundtrip_passed':True},
      'solver_diagnostic_missing_values':{str(k):int(v) for k,v in diagnostics.isna().sum().items() if v},
      'output_sha256':{name:sha256(OUT/name) for name in outputs}}
    (OUT/'preprocessing.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
    (OUT/'README.md').write_text(f'''# Preprocessed U-bend CFD data

This follows the original project's long-form CSV preparation: numeric inputs and physical-unit targets, stable case/row identifiers, and a separate fixed split file. Raw source files remain in `../../raw/ubend_cfd/`.

- `ubend_longform.csv`: **{len(model_data):,} successful simulations**, 28 input columns, pressure loss `DelP_Pa` and `CoolingObjective_K2m2`.
- `random90_10_split.csv`: original supplied assignments, **{config['rows']['train']:,} train / {config['rows']['test']:,} test**, joined by `case_uid`. No split was regenerated.
- `failed_simulations.csv`: **{len(failed):,} excluded unsuccessful runs**, original target sentinel values and failure reasons; preserved for a separate failure-detection study.
- `solver_diagnostics.csv`: last-iteration target mean/std values, excluded from model inputs. Missing auxiliary diagnostics are preserved rather than imputed.
- `numeric_summary.csv`: physical-unit feature/target distributions.
- `preprocessing.json`: feature/target lists, row counts, validation checks, source revision and exact hashes.

The 28 released `input_*` columns are the only model features. IDs, targets, source split, and solver statistics are not features. `geometry_id` identifies an individual design; it is not analogous to the six wavy-channel geometry families. Every retained input vector is unique. No Reynolds/Prandtl/Darcy columns are invented for this fixed-operating-condition dataset.

The cooling objective is the integrated squared temperature difference over the heated surface, not Nusselt number. The scalar source targets `target_0` and `target_1` are retained as the response values; solver-iteration means are auxiliary diagnostics. Both response columns remain in physical units. Signed geometry inputs are not logarithmically transformed. Scaling and any model-specific target transformation must be fitted/applied inside the training workflow, using fit data only. No scaler is fitted on the complete dataset.

All successful cases are retained; no imputation or error-based outlier filtering is performed. Pressure sentinel codes 10000/11000 are kept only in the failed-run file and cannot enter the regression labels. The full aggregate raw file overlaps the provided splits and must not be appended as additional observations.

Source revision: `{manifest['revision']}`. Source and citation/license details are preserved with the raw data. Reproduce from the project root with `python scripts/preprocess_ubend.py` (requires pandas, NumPy and pyarrow). Existing wavy-channel experiment runners still use their seven-feature schema; this preparation does not run or silently change those experiments.
''',encoding='utf-8')
    print(json.dumps({'output':str(OUT),'rows':config['rows'],'validation':config['validation']},indent=2))

if __name__=='__main__':main()
