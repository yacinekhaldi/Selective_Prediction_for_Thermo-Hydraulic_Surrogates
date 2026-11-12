"""Run the complete, resumable U-bend study with two isolated workers."""
import os
os.environ.setdefault('OMP_NUM_THREADS','3')
import contextlib,json,time,traceback
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
import ubend_experiments as U

def initialize():
    global DATAFRAME,THREAD_LIMIT
    import pandas as pd
    from threadpoolctl import threadpool_limits
    THREAD_LIMIT=threadpool_limits(limits=3)
    DATAFRAME=pd.read_csv(U.DATA/'ubend_longform.csv')
    DATAFRAME['group_id']=DATAFRAME.sample_id

def execute(job):
    seed,fold,target,model=job
    folder=U.OUT/'runs'/f'{seed}_{fold}_{target}_{model}'
    folder.mkdir(parents=True,exist_ok=True)
    with (folder/'execution.log').open('a',encoding='utf-8') as stream,contextlib.redirect_stdout(stream),contextlib.redirect_stderr(stream):
        U.one_experiment(DATAFRAME,*job)
    return job

def main():
    initialize();U.audit(DATAFRAME)
    jobs=[(s,f,t,m) for s in U.SEEDS for f in U.FOLDS for t in U.TARGETS for m in U.MODELS]
    todo=[j for j in jobs if not (U.OUT/'runs'/f'{j[0]}_{j[1]}_{j[2]}_{j[3]}'/'complete.json').exists()]
    U.save_json(dict(protocol='ubend_round1_v1',expected_runs=len(jobs),folds=U.FOLDS,seeds=U.SEEDS,models=U.MODELS,targets=U.TARGETS,workers=2,shift_definition='fixed input_0,input_6,input_12 low/high outer quintiles and central 40-60% slabs; two joint marginal-exclusion stress tests',primary_tolerances={'CoolingObjective_K2m2':5,'DelP_Pa':10},tolerances_are_illustrative=True,main_policy='unbounded UD_max, independent 90th percentile calibration; binary accept/abstain',physics='positivity and expanded fit-response range only; no assumed geometry monotonicity',data_preprocessing='identity geometry coordinates; fit-only standardization; pressure log1p as in original; no solver-statistic features'),U.OUT/'study_protocol.json')
    print(f'{len(jobs)} expected runs; {len(todo)} remaining.',flush=True)
    completed=len(jobs)-len(todo);started=time.monotonic()
    with ProcessPoolExecutor(max_workers=2,initializer=initialize) as pool:
        futures={pool.submit(execute,job):job for job in todo}
        for future in as_completed(futures):
            job=futures[future]
            try:future.result()
            except Exception:
                (U.OUT/'run_failure.txt').write_text(str(job)+'\n'+traceback.format_exc(),encoding='utf-8')
                raise
            completed+=1
            U.save_json(dict(completed=completed,expected=len(jobs),latest=job,elapsed_seconds=time.monotonic()-started),U.OUT/'progress.json')
            print(f'{completed}/{len(jobs)} '+str(job),flush=True)
    U.aggregate()
    print('All experiments complete; aggregates exported.',flush=True)

if __name__=='__main__':main()
