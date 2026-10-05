"""Plot recomputed OOF metrics and conditional patient-cluster intervals."""
import argparse
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

GROUPS=['previous_agnostic','previous_aware','previous_nomask',
        'presence_aware','presence_zero','patch_cap2048','patch_cap4096']
LABELS=['Prior agnostic','Prior aware','Prior no mask','Control mask',
        'Control zero mask','Patch 2048','Patch 4096']


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results-dir',type=Path,default=Path('results/smc_metric_review_20261005'))
    args=p.parse_args();root=args.results_dir
    ensemble=pd.read_csv(root/'ensemble_metrics.csv').set_index('group').loc[GROUPS]
    fig,axes=plt.subplots(1,3,figsize=(13,5),sharey=True,layout='constrained')
    colors=['#a0a6ac']*3+['#2878b5','#71a6d2','#cf8b26','#e5b875']
    y=np.arange(len(GROUPS))
    for ax,metric,title in zip(axes,['pr_auc','f1','mcc'],['Average precision','Positive-class F1 @ 0.5','MCC @ 0.5']):
        ax.barh(y,ensemble[metric],color=colors)
        for i,value in enumerate(ensemble[metric]): ax.text(value+.006,i,f'{value:.3f}',va='center',fontsize=9)
        ax.set_xlim(0,.5);ax.set_title(title);ax.set_yticks(y,LABELS);ax.invert_yaxis();ax.grid(axis='x',alpha=.2)
    fig.suptitle('ACR >= 2R: 575 events, 14 positive events, 133 patients | Five-seed OOF ensembles',fontsize=12)
    fig.savefig(root/'control_metric_comparison.png',dpi=180);plt.close(fig)
    seed=pd.read_csv(root/'paired_seed_deltas.csv')
    intervals=pd.read_csv(root/'paired_patient_bootstrap.csv')
    pairs=[('presence_aware','presence_zero','Zero mask minus observed mask'),
           ('patch_cap2048','patch_cap4096','4096 minus 2048 training patches')]
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    for ax,(left,right,title) in zip(axes,pairs):
        subset=seed[(seed.left==left)&(seed.right==right)]
        for i,metric in enumerate(['pr_auc','f1']):
            values=subset[f'delta_{metric}'].to_numpy()
            ax.scatter(np.full(5,i)+np.linspace(-.12,.12,5),values,color='#2878b5',s=32)
            for x,value,sv in zip(np.full(5,i)+np.linspace(-.12,.12,5),values,subset.seed):
                ax.annotate(str(sv),(x,value),xytext=(3,4),textcoords='offset points',fontsize=8)
            row=intervals[(intervals.left==left)&(intervals.right==right)&(intervals.metric==metric)].iloc[0]
            center=row.delta_right_minus_left
            ax.errorbar(i+.3,center,yerr=[[center-row.ci_low],[row.ci_high-center]],fmt='D',color='#cf8b26',capsize=4,label='Ensemble: patient CI' if i==0 else None)
        ax.axhline(0,color='gray',lw=1);ax.set_xticks([0,1],['AP change','F1 change']);ax.set_title(title,fontsize=11)
        ax.set_ylabel('Right minus left');ax.grid(axis='y',alpha=.2);ax.legend(fontsize=8)
    fig.suptitle('Dots: paired seeds | Diamonds: ensemble delta + conditional 95% patient-cluster interval',fontsize=11)
    fig.savefig(root/'paired_control_deltas.png',dpi=180);plt.close(fig)


if __name__=='__main__': main()
