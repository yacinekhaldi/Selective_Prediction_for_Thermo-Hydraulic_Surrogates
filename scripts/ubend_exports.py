"""Export U-bend tables, figures, plot data and a neutral standalone report."""
import json,re,subprocess
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import ubend_experiments as U

LABEL={'ExtraTrees_seed':'ET-seed','ExtraTrees_bootstrap':'ET-bag','XGBoost_seed':'XGB','GP_ARD':'GP-ARD','LogPolynomial':'Log-poly','CQR':'CQR'}
TARGET={'CoolingObjective_K2m2':'Cooling objective','DelP_Pa':'Pressure drop'}
SELECTED=['grouped_random','input_0_low','input_0_middle','input_0_high','input_6_high','combined_high']
METHOD={'U':'Uncertainty','D':'Distance','UD_max':'Max U+D','UD_equal':'Equal U+D','UDV_max':'Max U+D+V','UDV_equal':'Equal U+D+V','V':'Range/positivity','leverage':'Leverage','support_gap':'Coordinate gap','learned':'Learned gate','legacy_clipped':'Clipped reference','CQR_width':'CQR width','random':'Random'}
TABLES=U.OUT/'tables';FIGS=U.OUT/'figures';PLOTS=U.OUT/'plot_data';REPORTS=U.OUT/'reports'
CAPTIONS={}

def foldname(f):
    return {'grouped_random':'Design random','random_supplied':'Supplied random','combined_low':'Joint low','combined_high':'Joint high'}.get(f,f.replace('input_','x').replace('_',' '))

def esc(s):
    s=str(s)
    for a,b in [ ('\\',r'\textbackslash{}'),('&',r'\&'),('%',r'\%'),('_',r'\_'),('#',r'\#')]:s=s.replace(a,b)
    return s

def fmt(x):
    if pd.isna(x):return '--'
    if isinstance(x,(int,np.integer)):return str(x)
    if isinstance(x,(float,np.floating)):
        if not np.isfinite(x):return '--'
        return f'{x:.3g}' if abs(x)>=1e5 else f'{x:.3f}'
    return esc(x)

def table(name,frame,caption):
    frame.to_csv(TABLES/(name+'.csv'),index=False)
    long=len(frame)>28
    start=(r'\begingroup\small\setlength{\tabcolsep}{3pt}\begin{longtable}{' if long else r'\begin{table}[htbp]\centering\small\setlength{\tabcolsep}{3pt}\begin{tabular}{')+('l'*len(frame.columns))+'}\n'
    header=' & '.join(map(esc,frame.columns))+r'\\\midrule'+'\n'
    if long:
        start+=r'\caption{'+esc(caption)+r'}\label{tab:ubend-'+name+r'}\\\toprule'+'\n'+header+r'\endfirsthead\toprule'+'\n'+header+r'\endhead'+'\n'
    else:start+=r'\toprule'+'\n'+header
    body='\n'.join(' & '.join(fmt(x) for x in row)+r'\\' for row in frame.itertuples(index=False,name=None))
    end=r'\bottomrule\end{longtable}\endgroup' if long else r'\bottomrule\end{tabular}\caption{'+esc(caption)+r'}\label{tab:ubend-'+name+r'}\end{table}'
    (TABLES/(name+'.tex')).write_text(start+body+'\n'+end+'\n',encoding='utf-8')
    CAPTIONS[name]=caption

def savefig(name,fig,data=None):
    fig.savefig(FIGS/(name+'.pdf'),bbox_inches='tight')
    fig.savefig(FIGS/(name+'.png'),dpi=240,bbox_inches='tight')
    plt.close(fig)
    if data is not None:data.to_csv(PLOTS/(name+'.csv'),index=False)

def build_tables(m):
    source=pd.read_csv(U.DATA/'ubend_longform.csv')
    profile=source[U.FEATURES+U.TARGETS].describe().T.rename_axis('Variable').reset_index()
    table('dataset_summary',profile,'Successful U-bend design data in released coordinate/physical target units. Inputs are signed geometry parameters, pressure is Pa, and the cooling objective is K squared times m squared. No error-based trimming or global standardization is applied.')
    definitions=json.loads((U.OUT/'fold_definitions.json').read_text())
    counts=[]
    for entry in definitions:
        c=entry['counts'];used=sum(c[k] for k in ['fit','tune','scale','cal','test'])+(c['id_test'] if entry['fold']!='grouped_random' else 0)
        counts.append([foldname(entry['fold']),c['fit'],c['tune'],c['scale'],c['cal'],c['id_test'] if entry['fold'] not in ['random_supplied','grouped_random'] else 0,c['test'],len(source)-used])
    table('split_counts',pd.DataFrame(counts,columns=['Fold','Fit','Tune','Scale','Cal.','Separate ID','Test','Unused']),'Exact row counts, identical across outer seeds. Random tests use their test set as ID evaluation; structured tests have a separate supported ID set. Joint stress tests deliberately exclude unused marginal-only cases.')
    p=m[(m.role=='test')&(m.method=='UD_max')]
    keys=['fold','target','model'];cols=['all_mape','coverage','retained_mape','retained_failure','failure_prevalence','false_accept','false_abstain','auroc','auprc','spearman','aurc']
    mean=p.groupby(keys)[cols].mean().reset_index();sd=p.groupby(keys)[['all_mape','retained_mape']].std().reset_index()
    def readable(z):
        d=z.copy();d['Fold']=d.fold.map(foldname);d['Target']=d.target.map(TARGET);d['Model']=d.model.map(LABEL)
        d['All MAPE %']=d.all_mape;d['Coverage %']=100*d.coverage;d['Ret. MAPE %']=d.retained_mape;d['Ret. fail %']=100*d.retained_failure
        return d[['Fold','Target','Model','All MAPE %','Coverage %','Ret. MAPE %','Ret. fail %']]
    table('full_primary',readable(mean),'U-bend U+D-max results, means over three full runs. Empty accepted sets have undefined retained metrics. All error columns are percentages.')
    for fold,name in [('grouped_random','grouped'),('random_supplied','supplied_random')]:
        z=mean[mean.fold==fold].merge(sd,on=keys,suffixes=('','_sd'))
        d=readable(z).drop(columns='Fold');d['All SD']=z.all_mape_sd.to_numpy();d['Ret. SD']=z.retained_mape_sd.to_numpy()
        table(name,d,'Random-design evaluation, three-run means and sample SD. Calibration uses a nominal 90% acceptance budget; measured test coverage is shown.')
    table('structured',readable(mean[~mean.fold.isin(['grouped_random','random_supplied'])]),'Geometry-coordinate support shifts, defined before examining errors; low/high are endpoint support tests and middle is a missing interior slab. Means over three runs.')
    z=mean[mean.fold.isin(SELECTED)&mean.model.isin(['ExtraTrees_seed','XGBoost_seed','GP_ARD'])]
    decision=readable(z);decision['Abstained %']=100*(1-z.coverage.to_numpy())
    table('gate_decisions',decision,'Binary calibrated acceptance/abstention. The warning tier is a separate retrospective diagnostic only.')
    failure=z[['fold','target','model','failure_prevalence','false_accept','false_abstain']].copy()
    failure['fold']=failure.fold.map(foldname);failure['target']=failure.target.map(TARGET);failure['model']=failure.model.map(LABEL)
    for c in ['failure_prevalence','false_accept','false_abstain']:failure[c]*=100
    failure.columns=['Fold','Target','Model','Failures %','False accept %','False abstain %']
    table('false_decisions',failure,'False accept and false abstain use all test rows as denominator; failure prevalence is tolerance defined. Tolerances are illustrative, not engineering standards.')
    detection=mean[['fold','target','model','failure_prevalence','auroc','auprc','spearman']].copy()
    valid=p.groupby(keys).auroc.count().rename('Valid AUC runs').reset_index();detection=detection.merge(valid,on=keys)
    detection.fold=detection.fold.map(foldname);detection.target=detection.target.map(TARGET);detection.model=detection.model.map(LABEL)
    detection.failure_prevalence*=100;detection.columns=['Fold','Target','Model','Failures %','AUROC','AUPRC','Spearman','Valid AUC runs']
    table('detection',detection,'Failure detection means over evaluable runs; undefined single-class detection metrics remain blank/--. Valid AUC run count is explicit.')
    z=m[(m.role=='test')&(m.fold=='grouped_random')&(m.model=='XGBoost_seed')]
    z=z.groupby(['method','target'])[['coverage','retained_mape','aurc','auroc','auprc','spearman']].mean().reset_index()
    z.method=z.method.map(lambda x:METHOD.get(x,x));z.target=z.target.map(TARGET);z.coverage*=100
    z.columns=['Score','Target','Coverage %','Ret. MAPE %','AURC','AUROC','AUPRC','Spearman']
    table('ablation',z,'XGB score ablation on design-random tests. AURC is tie-aware and undefined for constant scores; it is a ranking diagnostic, not a risk guarantee.')
    ci=pd.read_csv(U.OUT/'cluster_intervals.csv');coarse=pd.read_csv(U.OUT/'coarse_cluster_intervals.csv');rows=[]
    for model in ['ExtraTrees_seed','XGBoost_seed','GP_ARD']:
        for target in U.TARGETS:
            a=ci[(ci.fold=='grouped_random')&(ci.model==model)&(ci.target==target)].iloc[0]
            b=coarse[(coarse.fold=='grouped_random')&(coarse.model==model)&(coarse.target==target)&(coarse.block=='geometry_cluster')].iloc[0]
            rows.append([LABEL[model],TARGET[target],int(a.n_clusters),f'[{100*a.coverage_lo:.1f}, {100*a.coverage_hi:.1f}]',f'[{a.retained_mape_lo:.3f}, {a.retained_mape_hi:.3f}]',int(b.n_clusters),f'[{b.retained_mape_lo:.3f}, {b.retained_mape_hi:.3f}]'])
    table('intervals',pd.DataFrame(rows,columns=['Model','Target','Designs','Coverage band %','Design risk band','Strata','Stratum risk band']),'Seed 42; 500 resamples conditional on fitted model. One row per unique design. Six covariate-only KMeans strata give coarse descriptive bands, not confidence intervals for unseen physical geometry families.')
    z=m[(m.role=='test')&(m.fold=='grouped_random')&m.method.isin(['UD_max','U','CQR_width'])]
    z=z.groupby(['model','target','method'])[['coverage','retained_mape','interval_coverage']].mean().reset_index();z.model=z.model.map(LABEL);z.target=z.target.map(TARGET);z.method=z.method.map(METHOD);z.coverage*=100;z.interval_coverage*=100
    z.columns=['Model','Target','Score','Coverage %','Ret. MAPE %','Interval coverage %']
    table('competitors',z,'Point-selection risk and CQR interval coverage are different estimands. GP pressure point estimates are inverse-log medians, while tree ensemble estimates are physical-unit member means.')
    matched=pd.read_csv(U.OUT/'matched.csv');idx=matched.groupby(['seed','fold','target','model','role','method']).coverage.apply(lambda x:(x-.9).abs().idxmin())
    matched90=matched.loc[idx.to_numpy()];matched90.to_csv(U.OUT/'matched90.csv',index=False)
    z=matched90[(matched90.role=='test')&(matched90.fold=='grouped_random')&matched90.method.isin(['U','D','UD_max','UD_equal'])]
    z=z.groupby(['model','target','method']).mape.mean().unstack('method').reset_index();z.model=z.model.map(LABEL);z.target=z.target.map(TARGET)
    z.columns=['Model','Target']+[METHOD[c] for c in z.columns[2:]]
    table('matched_risk',z,'MAPE at approximately 90% expected retained coverage, averaged over three runs. Random ordering inside ties may be unattainable by one deterministic cutoff.')
    for filename,name in [('binary_coverage_sweep','binary_sweep'),('tolerance_sweep','tolerances')]:
        z=pd.read_csv(U.OUT/(filename+'.csv'))
        z=z[(z.seed==42)&(z.model=='XGBoost_seed')]
        if 'role' in z:z=z[z.role=='test']
        param='calibration_quantile' if filename.startswith('binary') else 'tolerance'
        z=z[['fold','target',param,'coverage','retained_mape','retained_failure']].copy();z.fold=z.fold.map(foldname);z.target=z.target.map(TARGET);z.coverage*=100;z.retained_failure*=100
        z.columns=['Fold','Target',param,'Coverage %','Ret. MAPE %','Ret. fail %']
        table(name,z,'XGB seed-42 sensitivity. Binary calibration quantiles vary the cutoff; tolerance sweeps vary failure labels without refitting or moving the cutoff.')
    z=pd.read_csv(U.OUT/'conformal_group_coverage.csv').query("role == 'test'").groupby(['fold','target'])[['row_coverage','group_coverage','mean_width']].mean().reset_index();z.fold=z.fold.map(foldname);z.target=z.target.map(TARGET);z.row_coverage*=100;z.group_coverage*=100
    z.columns=['Fold','Target','Row coverage %','Design coverage %','Mean width']
    table('conformal_groups',z,'CQR 90% intervals, three-run means. Each design contains one row, so row/design coverage coincide. Group exchangeability is hypothetical under Sobol sampling and does not hold automatically for imposed support shifts. Width is in physical target units.')
    z=pd.read_csv(U.OUT/'point_baselines.csv').groupby(['fold','target','model'])[['n_fit','mape','rmse','r2']].mean().reset_index();z.fold=z.fold.map(foldname);z.target=z.target.map(TARGET);z.columns=['Fold','Target','Model','Fit rows','MAPE %','RMSE','R2']
    table('point_baselines',z,'Original five point-regressor families on full outer training. These are context benchmarks, not training-budget-matched selective models; RMSE is in target units.')
    z=pd.read_csv(U.OUT/'fit_budget_svr_controls.csv').groupby(['fold','target','model','fit_kind'])[['n_fit','mape']].mean().reset_index();z.fold=z.fold.map(foldname);z.target=z.target.map(TARGET);z.columns=['Fold','Target','Model','Fit budget','Fit rows','MAPE %']
    table('fit_budget',z,'Single ET on common fit versus full outer train and tune-selected converged SVR on common fit. Three-run means isolate training budget from selective scoring.')
    z=pd.read_csv(U.OUT/'sensitivity.csv');z=z[(z.seed==42)&(z.fold=='grouped_random')&(z.model=='XGBoost_seed')&(z.target=='CoolingObjective_K2m2')&(z.role=='test')]
    rows=[]
    for (a,b),g in z.groupby(['q_predict','q_warn']):
        records=g.set_index('state');r=records.loc['predict']
        rows.append([a,b,r.tau_predict,r.tau_warn,100*records.loc['predict','coverage'],100*records.loc['warn','coverage'],records.loc['predict','mape'],records.loc['warn','mape']])
    table('sensitivity',pd.DataFrame(rows,columns=['Lower q','Upper q','Lower cutoff','Upper cutoff','Predict %','Warn %','Predict MAPE %','Warn MAPE %']),'Three-state diagnostic for XGB cooling objective, seed 42/design-random. Primary decisions are binary; no intermediate risk ordering is presumed.')
    cutoffs=m[(m.seed==42)&(m.method=='UD_max')&(m.role=='test')][['fold','target','model','threshold','coverage','retained_mape']].copy()
    cutoffs.fold=cutoffs.fold.map(foldname);cutoffs.target=cutoffs.target.map(TARGET);cutoffs.model=cutoffs.model.map(LABEL);cutoffs.coverage*=100
    cutoffs.columns=['Fold','Target','Model','Cal. cutoff','Test coverage %','Ret. MAPE %']
    table('actual_thresholds',cutoffs,'Independent 90th-percentile binary calibration cutoffs and actual test retention, seed 42. Unbounded scores permit cutoffs above one; cutoffs are not error-probability thresholds.')
    distribution=pd.read_csv(U.OUT/'error_distribution.csv').query("role == 'test'")
    distribution=distribution.groupby(['fold','target','model'])[['median_ape','p95_ape','mean_smape','nonpositive_prediction_fraction']].mean().reset_index()
    distribution.fold=distribution.fold.map(foldname);distribution.target=distribution.target.map(TARGET);distribution.model=distribution.model.map(LABEL);distribution.nonpositive_prediction_fraction*=100
    distribution.columns=['Fold','Target','Model','Median APE %','95th APE %','Mean sMAPE %','Nonpositive %']
    table('error_distribution',distribution,'Error-distribution summaries averaged over three runs. Symmetric APE is bounded by 200%; raw per-run quantiles, maxima and numerical guard counts are exported separately.')
    return mean,matched90

def figures(m):
    plt.rcParams.update({'font.size':10,'axes.labelsize':11,'legend.fontsize':8,'axes.titlesize':11})
    p=m[(m.role=='test')&(m.method=='UD_max')]
    grouped=p[p.fold=='grouped_random'].groupby(['model','target'])[['coverage','retained_mape','all_mape']].mean().reset_index()
    fig,axes=plt.subplots(1,2,figsize=(9,3.8))
    for ax,target in zip(axes,U.TARGETS):
        for model,g in grouped[grouped.target==target].groupby('model'):ax.scatter(100*g.coverage,g.retained_mape,label=LABEL[model],s=45)
        ax.set(title=TARGET[target],xlabel='Retained coverage (%)',ylabel='Retained MAPE (%)');ax.grid(alpha=.2)
    axes[-1].legend(loc='best');fig.tight_layout();savefig('coverage',fig,grouped)
    curves=pd.read_csv(U.OUT/'curves.csv')
    for target in U.TARGETS:
        fig,axes=plt.subplots(3,2,figsize=(8,10))
        selected=[]
        for ax,fold in zip(axes.flat,SELECTED):
            for model in ['ExtraTrees_seed','XGBoost_seed','GP_ARD']:
                z=curves[(curves.fold==fold)&(curves.target==target)&(curves.model==model)&(curves.role=='test')&(curves.method=='UD_max')]
                row=m[(m.seed==42)&(m.fold==fold)&(m.target==target)&(m.model==model)&(m.role=='test')&(m.method=='UD_max')]
                if len(row) and pd.notna(row.aurc.iloc[0]):ax.plot(z.coverage,z.risk,label=LABEL[model]);selected.append(z)
            ax.set(title=foldname(fold),xlabel='Retained fraction',ylabel='MAPE (%)');ax.grid(alpha=.2)
        axes.flat[0].legend();fig.tight_layout();savefig('risk_'+target,fig,pd.concat(selected))
    fig,axes=plt.subplots(3,2,figsize=(8,9));collected=[]
    for i,target in enumerate(U.TARGETS):
        for j,model in enumerate(['ExtraTrees_seed','XGBoost_seed','GP_ARD']):
            d=pd.read_csv(U.OUT/'runs'/f'42_grouped_random_{target}_{model}'/'predictions.csv').query("role == 'test'")
            ax=axes[j,i];ax.scatter(d.truth,d.prediction,s=5,alpha=.3);a=min(d.truth.min(),d.prediction.min());b=max(d.truth.max(),d.prediction.max());ax.plot([a,b],[a,b],'k--',lw=.8)
            unit='Pa' if target=='DelP_Pa' else r'K$^2$m$^2$'
            ax.set(title=LABEL[model]+' / '+TARGET[target],xlabel='Reference ('+unit+')',ylabel='Prediction ('+unit+')');ax.grid(alpha=.2);collected.append(d[['case_uid','truth','prediction','ape']].assign(model=model,target=target))
    fig.tight_layout();savefig('grouped_parity',fig,pd.concat(collected))
    fig,axes=plt.subplots(1,2,figsize=(9,4));collected=[]
    for ax,target in zip(axes,U.TARGETS):
        for model in ['ExtraTrees_seed','XGBoost_seed','GP_ARD']:
            d=pd.read_csv(U.OUT/'runs'/f'42_input_0_high_{target}_{model}'/'predictions.csv').query("role == 'test'")
            ax.scatter(d.raw_knn,d.ape,s=4,alpha=.22,label=LABEL[model]);collected.append(d[['case_uid','raw_knn','ape']].assign(model=model,target=target))
        ax.set(title=TARGET[target]+' / x0 high',xlabel='Raw mean five-neighbor distance',ylabel='APE (%)');ax.grid(alpha=.2)
    axes[-1].legend();fig.tight_layout();savefig('distance_failure',fig,pd.concat(collected))
    mix=pd.read_csv(U.OUT/'mixed_shift.csv');fig,axes=plt.subplots(1,2,figsize=(9,4));z=mix[mix.fold=='input_0_high'].groupby(['target','model','ood_fraction']).false_accept.mean().reset_index()
    for ax,target in zip(axes,U.TARGETS):
        for model in ['ExtraTrees_seed','XGBoost_seed','GP_ARD']:
            g=z[(z.target==target)&(z.model==model)];ax.plot(g.ood_fraction,100*g.false_accept,'o-',label=LABEL[model])
        ax.set(title=TARGET[target]+' / x0 high',xlabel='Shifted-case fraction',ylabel='False accepts (% of all cases)');ax.grid(alpha=.2)
    axes[-1].legend();fig.tight_layout();savefig('mixture',fig,z)
    maps=pd.read_csv(U.OUT/'map_data.csv');fig,axes=plt.subplots(2,3,figsize=(9,5.5))
    for i,target in enumerate(U.TARGETS):
        for j,model in enumerate(['ExtraTrees_seed','XGBoost_seed','GP_ARD']):
            z=maps[(maps.target==target)&(maps.model==model)];pivot=z.pivot(index='input_0_bin',columns='input_6_bin',values='coverage')
            im=axes[i,j].imshow(pivot,vmin=0,vmax=1,origin='lower',cmap='viridis',aspect='auto');axes[i,j].set(title=LABEL[model]+' / '+TARGET[target],xlabel='x6 quintile bin',ylabel='x0 quintile bin')
    fig.tight_layout(rect=[0,0,.92,1]);cax=fig.add_axes([.94,.15,.015,.7]);fig.colorbar(im,cax=cax,label='Accepted fraction');savefig('map',fig,maps)
    importance=pd.read_csv(U.OUT/'feature_importance.csv');fig,axes=plt.subplots(1,2,figsize=(9,4));z=importance[importance.fold=='grouped_random']
    for ax,target in zip(axes,U.TARGETS):
        g=z[z.target==target].nlargest(10,'importance').sort_values('importance');ax.barh(g.feature.str.replace('input_','x'),g.importance);ax.set(title=TARGET[target],xlabel='Impurity importance (descriptive)')
    fig.tight_layout();savefig('feature_importance',fig,z)
    full=pd.read_csv(U.DATA/'ubend_longform.csv');fig,axes=plt.subplots(1,2,figsize=(9,4));data=[]
    for ax,target in zip(axes,U.TARGETS):
        bins=pd.qcut(full.input_0,10,labels=False);g=full.groupby(bins).agg(coordinate=('input_0','mean'),response=(target,'mean'),count=(target,'size')).reset_index(drop=True);ax.plot(g.coordinate,g.response,'o-');ax.set(title=TARGET[target],xlabel='x0 (bin mean)',ylabel=TARGET[target]+(' (Pa)' if target=='DelP_Pa' else r' (K$^2$m$^2$)'));ax.grid(alpha=.2);data.append(g.assign(target=target))
    fig.tight_layout();savefig('dataset',fig,pd.concat(data))
    fig,axes=plt.subplots(1,2,figsize=(9,4));data=[]
    for ax,target in zip(axes,U.TARGETS):
        d=pd.read_csv(U.OUT/'runs'/f'42_grouped_random_{target}_XGBoost_seed'/'predictions.csv').query("role == 'test'")
        for method in ['U','D','UD_max']:ax.hist(d[method],bins=35,histtype='step',density=True,label=METHOD[method])
        ax.axvline(d.threshold.iloc[0],color='k',ls='--',label='Max cutoff');ax.set(title=TARGET[target],xlabel='Unbounded normalized score',ylabel='Density');data.append(d[['case_uid','U','D','UD_max','threshold']].assign(target=target))
    axes[-1].legend();fig.tight_layout();savefig('scores',fig,pd.concat(data))

def reports(m,mean,matched90):
    grouped=mean[mean.fold=='grouped_random']
    lines=['# U-bend independent-system evaluation','', 'All 468 planned experiments are included: six models, thirteen designs, two physical-unit targets, three complete partition/training seeds. Models are fitted and calibrated separately for this system.','', '## Design-random primary results','', '| Model | Target | All MAPE % | Retained MAPE % | Coverage % | Retained failure % |','|---|---|---:|---:|---:|---:|']
    for r in grouped.itertuples():lines.append(f'| {LABEL[r.model]} | {TARGET[r.target]} | {r.all_mape:.3f} | {r.retained_mape:.3f} | {100*r.coverage:.1f} | {100*r.retained_failure:.1f} |')
    lines+=['','## Comparative findings and limits','']
    p=mean[~mean.fold.isin(['grouped_random','random_supplied'])]
    usable=p[p.coverage>0];improved=usable.retained_mape<usable.all_mape
    lines.append(f'Across {len(usable)} nonempty shifted model/target/fold mean results, calibrated U+D max reduces retained MAPE relative to all-case MAPE in {int(improved.sum())}; {int((p.coverage==0).sum())} shifted combinations have zero mean coverage. This descriptive count compares different accepted populations and is not a guarantee or significance test.')
    z=matched90[(matched90.role=='test')&matched90.method.isin(['U','D','UD_max'])].pivot_table(index=['fold','target','model'],columns='method',values='mape')
    z=z.dropna(subset=['U','D','UD_max']);better=(z.UD_max<z[['U','D']].min(axis=1)-1e-10);worse=(z.UD_max>z[['U','D']].min(axis=1)+1e-10)
    lines.append(f'At approximately 90% expected coverage, max fusion is better than both U-only and D-only in {int(better.sum())} of {len(z)} averaged comparisons, worse than the stronger component in {int(worse.sum())}, and tied in the remainder. Expected random tie breaking need not be attainable by a deterministic cutoff. Complete per-run results remain available.')
    for target in U.TARGETS:
        g=p[(p.model=='GP_ARD')&(p.target==target)]
        lines.append(f'GP-ARD {TARGET[target].lower()} on coordinate shifts: coverage ranges from {100*g.coverage.min():.1f}% to {100*g.coverage.max():.1f}%; retained MAPE among nonempty sets ranges from {g.retained_mape.min():.3f}% to {g.retained_mape.max():.3f}%. Read these jointly with all-case error and retained failure, not as a statement of universal shifted reliability.')
    lines+=['','The second system provides evidence beyond the wavy-channel grid, but it does not add methodological novelty to max fusion. The complete comparisons include both selective-error improvements and remaining failures at measured coverage.','', '## Protocol and interpretation','', '- Primary failure tolerances are illustrative: cooling objective 5%, pressure 10%; full sweeps are exported.','- Low/high support tests and middle-slab interpolation tests are on fixed coordinates x0, x6 and x12. They are not Reynolds shifts or transfer to independently defined physical geometry families. Fold boundaries use covariates only, not response errors.','- The supplied split has 8,051 outer-training / 895 test cases. Design-random and structured evaluations use independent fit/tune/scale/calibration roles. All models share assignments.','- Geometry coordinates are signed and unlogged; fit-only standardization is used. Pressure is modeled using log1p except the log-polynomial log(y) comparator. GP pressure points are inverse-log medians; ensembles use physical-unit member means.','- CQR intervals have no shifted selective-risk guarantee. One row per design makes row/group coverage identical; Sobol sampling does not establish exchangeable random groups.','- Positivity and expanded-fit-range checks are descriptive diagnostics. No unsupported monotonic response law is imposed on geometry coordinates.','- Bootstrap intervals condition on the fitted model and observed successful designs. Six covariate-only KMeans strata and coordinate bins are coarse descriptive blocks, not physical simulation batches or confidence intervals for unseen families.','- Failed mesh/solver runs remain excluded from regression; sentinel values are not reference pressures. This study does not evaluate failure detection on those runs.','- Calibration-fixed selective results and expected matched-coverage rankings answer different questions. Empty accepted sets have undefined risk; single-class detection metrics and constant-score discriminatory AURC are undefined.','- Full-outer-training point baselines have a different budget from selective models. Same-fit single/ensemble and outer-training controls expose that distinction.','', '## Export layout','', '- `runs/`: complete metadata, per-case predictions, assignments, calibration scores, per-run metrics, curves, matched coverage, sensitivity; CQR interval endpoints.','- `tables/`: measured CSV tables and article-ready LaTeX fragments.','- `figures/`: vector PDF and high-resolution PNG plots.','- `plot_data/`: exact records/aggregates used in every figure.','- `predictions/point_baselines.csv`: point-benchmark predictions; selective predictions remain in run folders.','- `models/`: saved seed-42/supplied-split point baseline estimators and settings; selective checkpoint estimators can be reproduced from code/settings and are not serialized.','- `reports/`: this assessment, protocol and figure/table PDFs.','', 'Reproduction from the project root: `python scripts/test_ubend.py`; `python scripts/run_ubend_study.py`; `python scripts/ubend_controls.py`; `python scripts/ubend_diagnostics.py`; `python scripts/validate_ubend_outputs.py`; `python scripts/ubend_exports.py`.','']
    provenance=json.loads((U.ROOT/'data/raw/ubend_cfd/download_manifest.json').read_text())
    lines+=['## Data provenance','',f"Source: JDecke/ubend-cfd, updated tabular release, pinned revision {provenance['revision']}.",'Original dataset article: Decke, Wuensch and Sick, Data in Brief 50 (2023), 109477; DOI 10.1016/j.dib.2023.109477.','The updated release declares CC-BY-NC-4.0. Source documentation and hashes are preserved under data/raw/ubend_cfd. Results concern 8,946 successful simulations, conditional on exclusion of failed mesh/solver runs.','Models are fitted and calibrated separately on U-bend. This is replication of the evaluation across systems, not transfer of the fitted wavy-channel predictor.','Reproduce with requirements-ubend.txt. PDF exports additionally require pdflatex, booktabs, longtable, graphicx, hyperref and lmodern.','']
    (REPORTS/'RESULTS_SUMMARY.md').write_text('\n'.join(lines),encoding='utf-8')
    descriptions={
      'coverage':'Design-random test: three-run mean retained MAPE versus measured coverage; the cutoff is calibrated independently at a nominal 90% budget. Models can return different populations.',
      'grouped_parity':'Seed-42 design-random test cases, same held-out designs across models. Dashed lines indicate exact agreement; axes show physical target units.',
      'distance_failure':'Seed-42 high-x0 geometry-support test: response APE versus raw mean five-neighbor distance. Distances alone do not establish response error.',
      'mixture':'Common-model mixtures of supported and high-x0 test designs, averaged over three runs. False accepts use all test cases as the denominator; tolerances are illustrative.',
      'map':'Seed-42 design-random accepted fractions in x0/x6 quintile cells, averaged over actual test cases and all remaining geometry coordinates. Cell counts are supplied; this is not a full-grid physical validity map.',
      'feature_importance':'Seed-42 ExtraTrees common-fit control on design-random data. Top ten impurity importances are descriptive, not causal physical effects.',
      'dataset':'Response means in ten x0 bins across successful designs, averaged over other geometry coordinates; lines connect bin means, not additional CFD observations.',
      'scores':'Seed-42 design-random XGB uncertainty, distance and fused score distributions. The dashed line is the independently calibrated max-score cutoff.',
      'risk_CoolingObjective_K2m2':'Cooling-objective seed-42 risk-coverage curves, separate folds, expected uniform ordering within score ties. These are retrospective rankings, not guaranteed deterministic threshold paths.',
      'risk_DelP_Pa':'Pressure-drop seed-42 risk-coverage curves under the same tie-aware, per-fold conventions; error is physical-unit MAPE in percent.'}
    U.save_json(descriptions,PLOTS/'figure_descriptions.json')
    (U.OUT/'README.md').write_text('# U-bend results\n\nSee [the complete assessment](reports/RESULTS_SUMMARY.md), [protocol](study_protocol.json), [validation](validation_report.json), and the `tables`, `figures`, `plot_data`, `runs`, `predictions`, and `models` folders. The two systems have separate model schemas and evaluation records.\n',encoding='utf-8')
    preamble=r'\documentclass[10pt]{article}\usepackage[a4paper,landscape,margin=12mm]{geometry}\usepackage{booktabs,longtable,graphicx,array,hyperref,lmodern}\setlength{\emergencystretch}{3em}\begin{document}'
    doc=preamble+'\n'+r'\section*{U-bend evaluation: measured tables}'+'\n'
    for name in CAPTIONS:doc+=r'\clearpage\section*{'+esc(name.replace('_',' '))+r'}\input{../tables/'+name+r'.tex}'+'\n'
    doc+=r'\end{document}'
    (REPORTS/'tables_compendium.tex').write_text(doc,encoding='utf-8')
    doc=r'\documentclass[11pt]{article}\usepackage[a4paper,margin=18mm]{geometry}\usepackage{graphicx,hyperref,lmodern}\begin{document}\section*{U-bend evaluation: figures}'+'\n'
    for name in sorted(p.stem for p in FIGS.glob('*.pdf')):
        doc+=r'\clearpage\section*{'+esc(name.replace('_',' '))+r'}\noindent '+esc(descriptions[name])+r'\begin{center}\includegraphics[width=\textwidth,height=.78\textheight,keepaspectratio]{../figures/'+name+r'.pdf}\end{center}'+'\n'
    doc+=r'\end{document}'
    (REPORTS/'figures_compendium.tex').write_text(doc,encoding='utf-8')
    doc=r'\documentclass[11pt]{article}\usepackage[a4paper,margin=19mm]{geometry}\usepackage{booktabs,graphicx,hyperref,lmodern}\setlength{\emergencystretch}{3em}\begin{document}\section*{U-bend independent-system evaluation}'+'\n'
    # Reuse the exact prose in the machine-readable report; omit its Markdown table.
    for line in lines:
        if line.startswith('|') or not line:continue
        if line.startswith('### '):doc+=r'\subsection*{'+esc(line[4:])+'}\n'
        elif line.startswith('## '):
            doc+=r'\subsection*{'+esc(line[3:])+'}\n'
            if 'Design-random primary results' in line:
                doc+=r'\begin{center}\small\begin{tabular}{llrrrr}\toprule Model & Target & All MAPE & Ret. MAPE & Ret. (\%) & Fail (\%)\\\midrule'+'\n'
                for r in grouped.itertuples():doc+=' & '.join([esc(LABEL[r.model]),'Cooling' if r.target=='CoolingObjective_K2m2' else 'Pressure',f'{r.all_mape:.3f}',f'{r.retained_mape:.3f}',f'{100*r.coverage:.1f}',f'{100*r.retained_failure:.1f}'])+r'\\'+'\n'
                doc+=r'\bottomrule\end{tabular}\end{center}'+'\n'
        elif line.startswith('Reproduction from the project root:'):
            doc+=r'\noindent Reproduction from the project root:\begin{quote}\small'+'\n'
            for command in re.findall(r'`([^`]+)`',line):
                interpreter,path=command.split(' ',1)
                doc+=r'\noindent\texttt{'+interpreter+r' }\nolinkurl{'+path+r'}\par'+'\n'
            doc+=r'\end{quote}'+'\n'
        elif line.startswith('Source:'):
            prefix,revision=line.rsplit(' ',1)
            doc+=esc(prefix)+r'\par\noindent\nolinkurl{'+revision.rstrip('.')+r'}.\par\medskip'+'\n'
        elif not line.startswith('# '):doc+=esc(line.removeprefix('- ')).replace('`','')+r'\par\medskip'+'\n'
    doc+=r'\end{document}'
    (REPORTS/'ubend_report.tex').write_text(doc,encoding='utf-8')
    results=[]
    for stem in ['tables_compendium','figures_compendium','ubend_report']:
        for _ in range(2):
            run=subprocess.run(['pdflatex','-interaction=nonstopmode','-halt-on-error',stem+'.tex'],cwd=REPORTS,capture_output=True,text=True,errors='replace')
            (REPORTS/(stem+'_build.txt')).write_text(run.stdout+'\n'+run.stderr,encoding='utf-8')
            assert run.returncode==0,stem
        log=(REPORTS/(stem+'.log')).read_text(encoding='utf-8',errors='replace')
        errors=re.findall(r'^!.*|Overfull .*|.*undefined references.*',log,re.M)
        results.append(dict(pdf=stem+'.pdf',layout_warnings=errors))
    U.save_json(results,REPORTS/'pdf_checks.json')

def main():
    for p in [TABLES,FIGS,PLOTS,REPORTS]:p.mkdir(exist_ok=True)
    assert len(list((U.OUT/'runs').glob('*/complete.json')))==468,'Finish all runs before exporting final comparisons'
    m=pd.read_csv(U.OUT/'metrics.csv');mean,matched=build_tables(m);figures(m);reports(m,mean,matched)
    U.save_json(dict(tables=CAPTIONS,figures=[p.name for p in FIGS.glob('*.pdf')],plot_data=[p.name for p in PLOTS.glob('*.csv')]),U.OUT/'export_manifest.json')
    print('U-bend tables, figures, plot data and standalone reports exported.',flush=True)

if __name__=='__main__':main()
