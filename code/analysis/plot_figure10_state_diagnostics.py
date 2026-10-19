#!/usr/bin/env python3
"""Plot accepted compact Figure10 time-cloud summaries; never open state arrays."""
from pathlib import Path
import argparse,hashlib,json,math,sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))
from representation_collapse.reporting import ExperimentReporter


def digest(raw):return hashlib.sha256(raw).hexdigest()


def main():
 p=argparse.ArgumentParser(description=__doc__)
 for key in ('analysis','analysis-sha256','acceptance','acceptance-sha256','output','run-id'):p.add_argument('--'+key,required=True)
 a=p.parse_args();raw=Path(a.analysis).read_bytes();accepted_raw=Path(a.acceptance).read_bytes()
 if digest(raw)!=a.analysis_sha256 or digest(accepted_raw)!=a.acceptance_sha256:raise ValueError('explicit input hash mismatch')
 accepted=json.loads(accepted_raw)
 if (accepted.get('kind')!='figure10_compact_trajectory_execution_acceptance_v1' or accepted.get('status')!='accepted'
     or accepted.get('analysis_result_sha256')!=a.analysis_sha256 or accepted.get('clouds_verified')!=21
     or accepted.get('scientific_collapse_claim') is not False):raise ValueError('independently accepted compact analysis required')
 result=json.loads(raw);out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
 identities={'analysis_sha256':a.analysis_sha256,'acceptance_sha256':a.acceptance_sha256,'renderer_sha256':digest(Path(__file__).read_bytes())}
 with ExperimentReporter(out/'reporting',project='representation-collapse',mode='offline',run_id=a.run_id,
     config={'kind':'descriptive_time_cloud_visualization','independent_model_seeds':1,'inferential_tests':False,
             'variance_ratio_display':'posthoc_display_of_frozen_window_measurements'},identities=identities) as reporter:
  import matplotlib
  matplotlib.use('Agg')
  import matplotlib.pyplot as plt
  plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
  fig,axes=plt.subplots(1,3,figsize=(10.2,3.7),layout='constrained');groups=['A','B','pooled'];summaries={}
  for group in groups:
   early=result['clouds'][group+'_early']['views'];late=result['clouds'][group+'_late']['views']
   summaries[group]={view:{'early_pr':early[view]['participation_ratio'],'late_pr':late[view]['participation_ratio'],
     'early_covariance_trace':early[view]['sample_covariance_trace'],'late_covariance_trace':late[view]['sample_covariance_trace'],
     'late_to_early_trace_ratio':late[view]['sample_covariance_trace']/early[view]['sample_covariance_trace']} for view in ['hidden','hidden_and_cell']}
  for ax,view,title in zip(axes[:2],['hidden','hidden_and_cell'],['a  Hidden-state dimension','b  Complete H+C dimension']):
   for i,group in enumerate(groups):
    d=summaries[group][view];ax.plot([i-.11,i+.11],[d['early_pr'],d['late_pr']],color='#777777',lw=1)
   for offset,which,color in [(-.11,'early_pr','#2166ac'),(.11,'late_pr','#b66b15')]:
    ax.scatter([i+offset for i in range(3)],[summaries[g][view][which] for g in groups],color=color,label='Early phases 1–8' if which=='early_pr' else 'Late phases 24–31',s=35,zorder=3)
   ax.set(title=title,ylabel='Participation ratio',xticks=range(3),xticklabels=['Motif A','Motif B','Pooled'],ylim=(.9,2.1));ax.grid(axis='y',alpha=.2)
  ratios=[summaries[g]['hidden_and_cell']['late_to_early_trace_ratio'] for g in groups]
  axes[2].bar(range(3),ratios,color=['#777777','#777777','#9a719c'],width=.55);axes[2].axhline(1,color='black',lw=.8,ls='--')
  axes[2].set(title='c  Complete H+C variance',ylabel='Late / early covariance trace',yscale='log',xticks=range(3),xticklabels=['Motif A','Motif B','Pooled'])
  axes[2].grid(axis='y',alpha=.2)
  for i,ratio in enumerate(ratios):axes[2].annotate(f'{ratio:.3f}',(i,ratio),xytext=(0,5),textcoords='offset points',ha='center',fontsize=9)
  axes[0].legend(loc='upper left',fontsize=8,frameon=False)
  fig.suptitle('GC-LSTM seed 0, epoch 7 · complete-state diagnostics\nOwn-cloud centering; correlated time samples; descriptive measurements',fontsize=12)
  files={}
  for suffix in ['pdf','png']:
   target=out/('figure10-state-diagnostics.'+suffix);fig.savefig(target,dpi=250);files[target.name]=digest(target.read_bytes())
  plt.close(fig)
  manifest={'kind':'figure10_state_summary_plot_v1','identities':identities,'summaries':summaries,'files':files,
    'caption':'Early phases 1–8 and late phases 24–31, excluding phase zero. Each per-motif window has 32 correlated targets; pooled windows have64. Each cloud is centered independently. Hidden and complete H+C PR answer different questions. Pooling also mixes motif means. Variance ratios are a posthoc display of the frozen covariance traces. One model seed/graph; no confidence intervals, causal memory or Floquet inference.',
    'model_seeds':1,'scientific_collapse_claim':False,'original_state_arrays_independently_reanalyzed':False}
  manifest_raw=(json.dumps(manifest,sort_keys=True,indent=2,allow_nan=False)+'\n').encode();(out/'figure-manifest.json').write_bytes(manifest_raw)
  for step,group in enumerate(groups):reporter.log({'plot/group':group,'plot/complete_trace_ratio':summaries[group]['hidden_and_cell']['late_to_early_trace_ratio']},step=step)
  reporter.log_manifest('figure10-state-summary-plot',metadata={'scientific_collapse_claim':False,'model_seeds':1},identities={**identities,'figure_manifest_sha256':digest(manifest_raw)})
 print(json.dumps({'output':str(out),'scientific_collapse_claim':False}))


if __name__=='__main__':main()
