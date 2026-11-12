"""Download a pinned snapshot of the public U-bend tabular dataset."""
import hashlib,json,ssl,urllib.request
from datetime import datetime,timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
DEST=ROOT/'data/raw/ubend_cfd'
REPO='JDecke/ubend-cfd'

def validate():
    import numpy as np
    import pandas as pd
    import pyarrow
    manifest=json.loads((DEST/'download_manifest.json').read_text())
    for f in manifest['files']:
        assert hashlib.sha256((DEST/f['path']).read_bytes()).hexdigest()==f['sha256']
    frames={n:pd.read_parquet(DEST/'params'/f'{n}.parquet') for n in ['data','train','test','failed']}
    inputs=[f'input_{i}' for i in range(28)]
    ids={n:set(frames[n].sample_id) for n in ['train','test','failed']}
    assert all(not ids[a]&ids[b] for a,b in [('train','test'),('train','failed'),('test','failed')])
    assert set.union(*ids.values())==set(range(len(frames['data'])))
    for n,d in frames.items():
        assert np.isfinite(d[inputs+['target_0','target_1']].to_numpy()).all()
        if n!='data':
            actual=frames['data'].iloc[d.sample_id.to_numpy()].reset_index(drop=True)
            pd.testing.assert_frame_equal(actual,d[actual.columns].reset_index(drop=True))
    assert (frames['train'].target_0<10000).all() and (frames['test'].target_0<10000).all()
    assert frames['failed'].target_0.isin([10000,11000]).all()
    report=dict(status='passed',revision=manifest['revision'],rows={n:len(d) for n,d in frames.items()},inputs=inputs,main_targets=['target_0','target_1'],columns={n:d.columns.tolist() for n,d in frames.items()},successful_cases=len(frames['train'])+len(frames['test']),failure_codes={str(int(k)):int(v) for k,v in frames['failed'].target_0.value_counts().items()},sha256_verified=True,splits_disjoint=True,split_union_matches_aggregate=True,split_values_match_aggregate_by_sample_id=True,parquet_reader_version=pyarrow.__version__)
    (DEST/'validation.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    note=DEST/'DATASET_README.md'
    text=note.read_text(encoding='utf-8').split('\n## Verified local counts')[0]
    text+='\n## Verified local counts\n\n'+f"Aggregate: {report['rows']['data']:,} cases. Supplied train: {report['rows']['train']:,}; test: {report['rows']['test']:,}; failed: {report['rows']['failed']:,}. The {report['successful_cases']:,} successful cases and failed cases partition the aggregate without overlapping IDs. Split values exactly match the aggregate by sample ID. All 28 inputs and both main targets are finite.\n"
    note.write_text(text,encoding='utf-8')
    print(json.dumps({k:report[k] for k in ['status','rows','successful_cases','failure_codes','splits_disjoint']},indent=2))

def main():
    context=ssl.create_default_context()
    # Accommodate the local Windows certificate chain's missing AKI; certificate
    # authority and hostname verification remain enabled.
    context.verify_flags &= ~ssl.VERIFY_X509_STRICT
    def fetch(url):
        request=urllib.request.Request(url,headers={'User-Agent':'U-bend-research-dataset-download/1.0'})
        with urllib.request.urlopen(request,context=context,timeout=90) as r:return r.read()
    DEST.mkdir(parents=True,exist_ok=True)
    manifest_path=DEST/'download_manifest.json'
    revision=json.loads(manifest_path.read_text())['revision'] if manifest_path.exists() else json.loads(fetch(f'https://huggingface.co/api/datasets/{REPO}/revision/main'))['sha']
    files=['README.md','params/data.parquet','params/train.parquet','params/test.parquet','params/failed.parquet']
    records=[]
    for name in files:
        url=f'https://huggingface.co/datasets/{REPO}/resolve/{revision}/{name}'
        path=DEST/name;path.parent.mkdir(parents=True,exist_ok=True)
        data=fetch(url)
        if name.endswith('.parquet'):
            assert data[:4]==b'PAR1' and data[-4:]==b'PAR1',name
        path.write_bytes(data)
        records.append(dict(path=name,url=url,bytes=len(data),sha256=hashlib.sha256(data).hexdigest()))
        print(f'Downloaded {name}: {len(data):,} bytes',flush=True)
    manifest=dict(repository=f'https://huggingface.co/datasets/{REPO}',revision=revision,downloaded_at_utc=datetime.now(timezone.utc).isoformat(),configuration='params',license='CC-BY-NC-4.0 (as declared in upstream README)',files=records,scope='All tabular Parquet files and upstream documentation; full CFD fields and animations excluded.')
    manifest_path.write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    (DEST/'DATASET_README.md').write_text(f'''# U-bend CFD — local raw dataset

Source: https://huggingface.co/datasets/{REPO}
Pinned revision: `{revision}`
Original article: https://doi.org/10.1016/j.dib.2023.109477
License declared by this updated dataset release: **CC-BY-NC-4.0**. See the preserved upstream `README.md` for attribution and version details.

The original downloaded bytes are preserved under `params/`: `data.parquet`, `train.parquet`, `test.parquet`, and `failed.parquet`. The full CFD fields/animations are not needed for tabular surrogate evaluation and are not downloaded.

There are 28 geometry input parameters (`input_0` through `input_27`) and two main outputs: pressure loss (`target_0`) and a cooling-performance objective (`target_1`, integrated squared temperature deviation, not Nusselt number). Solver-iteration target means and standard deviations are auxiliary outputs, not geometry input features.

Failed runs use pressure-target sentinel values 10000 (mesh failure) and 11000 (solver divergence). Preserve these failures separately; do not treat sentinels as physical regression targets. Check sample IDs and overlap before using `data.parquet` alongside the supplied splits: the aggregate and split files are not independent datasets.

`download_manifest.json` records exact source URLs, revision, file sizes and SHA-256 checksums. `validation.json`, when present, records actual local schema and row counts. No model experiments have been run on this dataset yet.

To download this same pinned revision again from the project root: `python scripts/download_ubend.py`.
''',encoding='utf-8')
    print('Saved pinned tabular dataset to',DEST,flush=True)

if __name__=='__main__':
    import sys
    if '--validate-only' in sys.argv:validate()
    else:
        main()
        try:import pyarrow
        except ImportError:print('Install pyarrow to run --validate-only for row/schema verification.')
        else:validate()
