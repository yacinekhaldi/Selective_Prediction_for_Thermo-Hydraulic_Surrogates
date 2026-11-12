"""Export measured wavy-channel tables and figures."""
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from revision_experiments import ROOT,OUT,DATA
TABLES=OUT/'tables'
FIGS=OUT/'figures'
LABEL={'ExtraTrees_seed':'ET-seed','ExtraTrees_bootstrap':'ET-bag','XGBoost_seed':'XGB','GP_ARD':'GP-ARD','LogPolynomial':'Log-poly','CQR':'CQR'}
TL={'Nuavg':r'$Nu_{\rm avg}$','DelP_Pa':r'$\Delta P$'}
FL={'random_supplied':'Supplied random','grouped_random':'Grouped random','Re25':r'$Re=25$','Re100':r'$Re=100$','Re250':r'$Re=250$','Re500':r'$Re=500$','combined':'Combined'}

def esc(s):
    s=str(s)
    return s.replace('_',r'\_').replace('%',r'\%').replace('&',r'\&').replace('#',r'\#')

def num(v,d=2):return '--' if not np.isfinite(v) else f'{v:.{d}f}'

def table(name,columns,rows,caption,label,long=False):
    TABLES.mkdir(exist_ok=True)
    if long:
        text=r'\color{\revisioncolor}\scriptsize\begin{longtable}{'+('l'*len(columns))+ '}\n'+r'\caption{'+caption+r'}\label{'+label+r'}\\'+'\n'+r'\toprule'+'\n'+' & '.join(columns)+r'\\\midrule\endfirsthead'+'\n'+' & '.join(columns)+r'\\\midrule\endhead'+'\n'
        text+='\n'.join(' & '.join(map(str,row))+r'\\' for row in rows)+ '\n'+r'\bottomrule\end{longtable}'+'\n'
    else:
        text=r'\begin{table}[htbp]\color{\revisioncolor}\centering\scriptsize\caption{'+caption+r'}\label{'+label+'}\n'+r'\begin{tabular}{'+('l'*len(columns))+r'}\toprule'+'\n'+' & '.join(columns)+r'\\\midrule'+'\n'
        text+='\n'.join(' & '.join(map(str,row))+r'\\' for row in rows)+'\n'+r'\bottomrule\end{tabular}\end{table}'+'\n'
    (TABLES/f'{name}.tex').write_text(text,encoding='utf-8')

def figures(m):
    FIGS.mkdir(exist_ok=True)
    plt.rcParams.update({'font.size':11,'axes.labelsize':12,'legend.fontsize':9,'savefig.dpi':300})
    df=pd.read_csv(DATA/'02_processed_data/ML_dataset_longform.csv')
    fig,ax=plt.subplots(1,2,figsize=(10,3.6))
    for i,t in enumerate(['Nuavg','DelP_Pa']):
        for g,z in df.groupby('geometry_id'):
            q=z.groupby('Re')[t].mean();ax[i].plot(q.index,q.values,'o-',label=g)
        ax[i].set(xlabel=r'$Re$',ylabel=TL[t]+(' (Pa)' if i else ''));ax[i].grid(alpha=.2)
    ax[1].legend(ncol=2);fig.tight_layout();fig.savefig(FIGS/'dataset.pdf');plt.close(fig)
    curves=pd.read_csv(OUT/'curves.csv')
    for target in ['Nuavg','DelP_Pa']:
        fig,axes=plt.subplots(3,2,figsize=(8,10))
        for ax,fold in zip(axes.flat,['grouped_random','Re100','Re250','Re500','G04','combined']):
            for model in ['ExtraTrees_seed','XGBoost_seed','GP_ARD']:
                z=curves[(curves.fold==fold)&(curves.target==target)&(curves.model==model)&(curves.role=='test')&(curves.method=='UD_max')]
                status=m[(m.seed==42)&(m.fold==fold)&(m.target==target)&(m.model==model)&(m.role=='test')&(m.method=='UD_max')]
                if len(status) and pd.isna(status.aurc.iloc[0]):continue
                ax.plot(z.coverage,z.risk,label=LABEL[model])
            ax.set(title=FL.get(fold,fold),xlabel='Retained fraction',ylabel='MAPE (%)');ax.grid(alpha=.2)
        axes.flat[0].legend();fig.tight_layout();fig.savefig(FIGS/f'risk_{target}.pdf');plt.close(fig)
    z=m[(m.method=='UD_max')&(m.role=='test')]
    fig,ax=plt.subplots(1,2,figsize=(11,4))
    for i,target in enumerate(['Nuavg','DelP_Pa']):
        for model in ['ExtraTrees_seed','XGBoost_seed','GP_ARD','LogPolynomial']:
            g=z[(z.target==target)&(z.model==model)].groupby('fold').mean(numeric_only=True)
            folds=['grouped_random','Re100','Re250','Re25','Re500','G04','combined']
            ax[i].plot(range(len(folds)),[g.loc[f,'coverage']*100 for f in folds],'o-',label=LABEL[model])
        ax[i].set_xticks(range(len(folds)),['Grouped','Re100','Re250','Re25','Re500','G04','Combined'],rotation=35)
        ax[i].set(ylabel='Retained cases (%)',title=TL[target]);ax[i].grid(alpha=.2)
    ax[1].legend();fig.tight_layout();fig.savefig(FIGS/'coverage.pdf');plt.close(fig)
    mix=pd.read_csv(OUT/'mixed_shift.csv')
    fig,ax=plt.subplots(1,2,figsize=(10,4))
    for i,target in enumerate(['Nuavg','DelP_Pa']):
        for model in ['ExtraTrees_seed','XGBoost_seed','GP_ARD']:
            d=mix[(mix.fold=='Re250')&(mix.target==target)&(mix.model==model)].groupby('ood_fraction').mean(numeric_only=True)
            ax[i].plot(d.index,d.false_accept*100,'o-',label=LABEL[model])
        ax[i].set(xlabel='OOD fraction in test mixture',ylabel='False accepts (% of all cases)',title=TL[target]);ax[i].grid(alpha=.2)
    ax[1].legend();fig.tight_layout();fig.savefig(FIGS/'mixture.pdf');plt.close(fig)
    maps=pd.read_csv(OUT/'map_data.csv')
    fig,axes=plt.subplots(2,3,figsize=(11,6))
    for i,target in enumerate(['Nuavg','DelP_Pa']):
        for j,model in enumerate(['ExtraTrees_seed','XGBoost_seed','GP_ARD']):
            d=maps[(maps.target==target)&(maps.model==model)].pivot(index='Re',columns='Da',values='coverage')
            im=axes[i,j].imshow(d.values,origin='lower',vmin=0,vmax=1,cmap='viridis',aspect='auto')
            axes[i,j].set_xticks(range(len(d.columns)),[f'{v:.0e}' for v in d.columns],rotation=30)
            axes[i,j].set_yticks(range(len(d.index)),d.index);axes[i,j].set(xlabel=r'$Da$',ylabel=r'$Re$',title=LABEL[model]+' '+TL[target])
    fig.tight_layout(rect=[0,0,.91,1]);cax=fig.add_axes([.93,.15,.015,.7]);fig.colorbar(im,cax=cax,label='Accepted fraction');fig.savefig(FIGS/'map.pdf');plt.close(fig)
    # Unbounded versus clipped scores: use same cases and transparent binning.
    fig,ax=plt.subplots(1,2,figsize=(10,3.7))
    for i,fold in enumerate(['grouped_random','Re250']):
        p=OUT/'runs'/f'42_{fold}_Nuavg_XGBoost_seed/predictions.csv';d=pd.read_csv(p);d=d[d.role=='test']
        for col,label in [('UD_max','Unbounded U+D'),('legacy_clipped','Clipped reference')]:ax[i].hist(d[col],bins=30,histtype='step',density=True,label=label)
        ax[i].set(title=FL.get(fold,fold),xlabel='Reliability score',ylabel='Density');ax[i].legend()
    fig.tight_layout();fig.savefig(FIGS/'scores.pdf');plt.close(fig)

def build_tables(m):
    primary=m[(m.method=='UD_max')&(m.role=='test')]
    rows=[]
    for model in LABEL:
        for t in TL:
            z=primary[(primary.fold=='grouped_random')&(primary.model==model)&(primary.target==t)]
            mean=lambda col: num(z[col].mean())
            spread=lambda col:num(z[col].mean(),3)+r'$\pm$'+num(z[col].std(),3)
            rows.append([LABEL[model],TL[t],spread('all_mape'),num(z.coverage.mean()*100),spread('retained_mape'),num(z.retained_failure.mean()*100)])
    table('grouped',['Model','Target','All MAPE','Retained (\\%)','Retained MAPE','Failure (\\%)'],rows,'Group-disjoint random test results across three complete runs. MAPE values are percentages; spreads are sample standard deviations across outer seeds. Failure is conditional on retention.','tab:grouped')
    rows=[]
    for fold in ['Re25','Re100','Re250','Re500','G01','G02','G03','G04','G05','G06','combined']:
        for t in TL:
            row=[FL.get(fold,fold),TL[t]]
            for model in ['ExtraTrees_seed','XGBoost_seed','GP_ARD']:
                z=primary[(primary.fold==fold)&(primary.target==t)&(primary.model==model)]
                row.extend([num(z.all_mape.mean()),num(z.coverage.mean()*100)])
            rows.append(row)
    table('structured',['Fold','Target','ET MAPE','ET ret.','XGB MAPE','XGB ret.','GP MAPE','GP ret.'],rows,'Structured-fold all-case MAPE (\\%) and retained fraction (\\%), averaged across three training/partition seeds. ET denotes ET-seed. Interior Reynolds holdouts are interpolation tests.','tab:structured')
    rows=[]
    for method in ['U','D','V','UD_max','UD_equal','UDV_max','UDV_equal','leverage','support_gap','learned','legacy_clipped','random']:
        for t in TL:
            z=m[(m.fold=='grouped_random')&(m.model=='XGBoost_seed')&(m.target==t)&(m.role=='test')&(m.method==method)]
            if len(z):rows.append([esc(method),TL[t],num(z.coverage.mean()*100),num(z.retained_mape.mean()),num(z.aurc.mean()),num(z.auroc.mean(),3),num(z.auprc.mean(),3),num(z.spearman.mean(),3)])
    table('ablation',['Score','Target','Ret. (\\%)','Risk','AURC','AUROC','AUPRC',r'$\rho$'],rows,'XGBoost score comparisons on grouped random tests (means over three seeds). Risk and AURC use MAPE in percent. Detection metrics use the stated engineering failure threshold; undefined values are marked --. Complete fold/model results are supplied as CSV.','tab:ablation')
    ci=pd.read_csv(OUT/'cluster_intervals.csv');coarse_path=OUT/'coarse_cluster_intervals.csv';coarse=pd.read_csv(coarse_path) if coarse_path.exists() else None;rows=[]
    for model in ['ExtraTrees_seed','XGBoost_seed','GP_ARD']:
        for t in TL:
            z=ci[(ci.model==model)&(ci.target==t)&(ci.fold=='grouped_random')].iloc[0]
            geometry='--'
            if coarse is not None:
                g=coarse[(coarse.model==model)&(coarse.target==t)&(coarse.fold=='grouped_random')&(coarse.block=='geometry_id')].iloc[0]
                geometry=f'[{g.retained_mape_lo:.3f}, {g.retained_mape_hi:.3f}]'
            rows.append([LABEL[model],TL[t],int(z.n_clusters),f'[{z.coverage_lo*100:.1f}, {z.coverage_hi*100:.1f}]',f'[{z.retained_mape_lo:.3f}, {z.retained_mape_hi:.3f}]',geometry])
    table('intervals',['Model','Target','Tuples','Retained (\\%) CI','Tuple risk CI','Geometry risk CI'],rows,'Conditional 95\\% intervals from 500 bootstrap samples for seed 42: non-Pr tuples versus six whole geometry clusters. Risk is retained MAPE (\\%). Additional Reynolds-cluster intervals and insufficient-cluster flags are supplied; these descriptive intervals do not establish transfer to new physical families.','tab:intervals')
    rows=[]
    for t in TL:
        for model in ['ExtraTrees_seed','XGBoost_seed','GP_ARD','CQR']:
            for method in ['UD_max']+(['CQR_width'] if model=='CQR' else []):
                z=m[(m.target==t)&(m.model==model)&(m.fold=='grouped_random')&(m.role=='test')&(m.method==method)]
                rows.append([LABEL[model],TL[t],esc(method),num(z.coverage.mean()*100),num(z.retained_mape.mean()),num(z.get('interval_coverage',pd.Series(dtype=float)).mean()*100)])
    table('competitors',['Model','Target','Score','Ret. (\\%)','Ret. MAPE','Interval cov. (\\%)'],rows,'Same-data selective comparisons. CQR interval coverage is a different estimand from retained risk and is reported only for that model. GP uncertainty uses its predictive standard deviation.','tab:competitors')
    # Complete primary per-fold values, including metric availability and between-run variability.
    rows=[]
    for (fold,t,model),z in primary.groupby(['fold','target','model'],sort=True):
        rows.append([FL.get(fold,fold),TL[t],LABEL[model],num(z.all_mape.mean()),num(z.coverage.mean()*100),num(z.retained_mape.mean()),num(z.retained_failure.mean()*100),num(z.auroc.mean(),3),num(z.auprc.mean(),3)])
    table('full_primary',['Fold','Target','Model','All MAPE','Ret. (\\%)','Ret. MAPE','Fail (\\%)','AUROC','AUPRC'],rows,'Complete U+D max-gate primary fold/model results, averaged over three outer seeds. Undefined retained metrics at zero coverage are --; individual runs and all score ablations are provided in the accompanying CSV files.','tab:full',long=True)
    sensitivity=pd.read_csv(OUT/'sensitivity.csv');rows=[]
    d=sensitivity[(sensitivity.seed==42)&(sensitivity.fold=='grouped_random')&(sensitivity.target=='Nuavg')&(sensitivity.model=='XGBoost_seed')&(sensitivity.role=='test')]
    for (q1,q2),z in d.groupby(['q_predict','q_warn']):
        pred=z[z.state=='predict'].iloc[0];warn=z[z.state=='warn'].iloc[0];ab=z[z.state=='abstain'].iloc[0]
        rows.append([num(q1),num(q2),num(pred.tau_predict,3),num(pred.tau_warn,3),num(pred.coverage*100,1),num(warn.coverage*100,1),num(ab.coverage*100,1),num(pred.mape,3),num(warn.mape,3)])
    table('sensitivity',['Lower q','Upper q',r'$\tau_p$',r'$\tau_w$','Pred. \\%','Warn \\%','Abs. \\%','Pred. risk','Warn risk'],rows,'Diagnostic two-threshold sensitivity for XGBoost, grouped random Nusselt test, seed 42. The deployed policy is binary; the three-state experiment is retained to assess the intermediate-tier hypothesis. Full sensitivity results cover every model and fold.','tab:sensitivity')

def full_figures(P):
    out=P.OUT
    # Descriptive model diagnostics and parity panels, restoring the visual
    # analysis of the original article using the new grouped predictions.
    fig,axes=plt.subplots(2,3,figsize=(11,7))
    for i,target in enumerate(P.TL):
        for j,model in enumerate(['ExtraTrees_seed','XGBoost_seed','GP_ARD']):
            d=pd.read_csv(out/'runs'/f'42_grouped_random_{target}_{model}'/'predictions.csv');d=d[d.role=='test']
            ax=axes[i,j]
            for accepted,label,color in [(True,'Accepted','tab:blue'),(False,'Abstained','tab:orange')]:
                z=d[d.accepted==accepted];ax.scatter(z.truth,z.prediction,s=9,alpha=.55,label=label,color=color)
            low=min(d.truth.min(),d.prediction.min());high=max(d.truth.max(),d.prediction.max());ax.plot([low,high],[low,high],'k--',lw=1)
            ax.set(title=P.LABEL[model]+' '+P.TL[target],xlabel='Released reference'+(' (Pa)' if i else ''),ylabel='Prediction'+(' (Pa)' if i else ''));ax.grid(alpha=.15)
    axes[0,0].legend();fig.tight_layout();fig.savefig(P.FIGS/'grouped_parity.pdf');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for i,target in enumerate(P.TL):
        for model in ['ExtraTrees_seed','XGBoost_seed','GP_ARD']:
            d=pd.read_csv(out/'runs'/f'42_Re250_{target}_{model}'/'predictions.csv');d=d[d.role=='test']
            axes[i].scatter(d.raw_knn,d.ape,s=8,alpha=.25,label=P.LABEL[model])
        axes[i].set(xlabel='Mean five-neighbor distance',ylabel='Absolute percentage error (%)',title=P.TL[target]);axes[i].grid(alpha=.15)
    axes[1].legend();fig.tight_layout();fig.savefig(P.FIGS/'distance_failure.pdf');plt.close(fig)
    importance=pd.read_csv(out/'feature_importance.csv');fig,axes=plt.subplots(1,2,figsize=(10,4))
    for i,target in enumerate(P.TL):
        d=importance[(importance.fold=='grouped_random')&(importance.target==target)].sort_values('importance')
        labels={'Re':r'$Re$','Pr':r'$Pr$','Da':r'$Da$','epsi':r'$\epsilon$','Hp_mm':r'$H_p$','a_mm':r'$a$','Lw_mm':r'$L_w$'}
        axes[i].barh([labels[f] for f in d.feature],d.importance);axes[i].set(title=P.TL[target],xlabel='Impurity-based importance')
    fig.tight_layout();fig.savefig(P.FIGS/'feature_importance.pdf');plt.close(fig)
def main():
    TABLES.mkdir(exist_ok=True);FIGS.mkdir(exist_ok=True)
    m=pd.read_csv(OUT/'metrics.csv');build_tables(m);figures(m);full_figures(sys.modules[__name__])
    for p in TABLES.glob('*.tex'):
        p.write_text(p.read_text(encoding='utf-8').replace(r'\color{\revisioncolor}',''),encoding='utf-8')
    print('Wavy-channel scientific tables and figures exported.')
if __name__=='__main__':main()
