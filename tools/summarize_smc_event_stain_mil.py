#!/usr/bin/env python3
"""Summarize repeated event-MIL CV and create a five-seed OOF ensemble."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score

TASKS = ("acr_high", "amr_positive", "significant_rejection")
def arguments() -> argparse.Namespace:
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--results-root",type=Path,required=True); p.add_argument("--output-dir",type=Path,required=True); p.add_argument("--seeds",nargs="+",type=int,default=[1,11,21,31,41]); p.add_argument("--threshold",type=float,default=.5); return p.parse_args()
def metrics(frame: pd.DataFrame, threshold: float) -> dict[str,float|int]:
    y=frame.label.to_numpy(int); p=frame.probability.to_numpy(float); tn,fp,fn,tp=confusion_matrix(y,p>=threshold,labels=[0,1]).ravel()
    return {"events":len(y),"positive_events":int(y.sum()),"auroc":float(roc_auc_score(y,p)),"pr_auc":float(average_precision_score(y,p)),"sensitivity":float(tp/(tp+fn)) if tp+fn else float("nan"),"specificity":float(tn/(tn+fp)) if tn+fp else float("nan"),"balanced_accuracy":float(((tp/(tp+fn))+(tn/(tn+fp)))/2) if tp+fn and tn+fp else float("nan")}
def main() -> int:
    args=arguments(); args.output_dir.mkdir(parents=True,exist_ok=True); per_seed=[]; final=[]
    for task in TASKS:
        frames=[]
        for seed in args.seeds:
            path=args.results_root/f"{task}_l0_0p25mpp_seed{seed}"/"oof_predictions.csv"
            if not path.is_file(): raise FileNotFoundError(path)
            frame=pd.read_csv(path,dtype={"event_id":str,"case_id":str}); frame["seed"]=seed; frames.append(frame); per_seed.append({"task":task,"seed":seed,**metrics(frame,args.threshold)})
        combined=pd.concat(frames,ignore_index=True)
        if (combined.groupby("event_id").label.nunique()>1).any(): raise ValueError(f"{task}: inconsistent event labels across seeds")
        ensemble=combined.groupby("event_id",as_index=False).agg(case_id=("case_id","first"),label=("label","first"),probability=("probability","mean"),seed_predictions=("seed","nunique"))
        if not ensemble.seed_predictions.eq(len(args.seeds)).all(): raise ValueError(f"{task}: incomplete seed coverage")
        ensemble["prediction"]=(ensemble.probability>=args.threshold).astype(int); ensemble.to_csv(args.output_dir/f"{task}_seed_ensemble_oof.csv",index=False)
        seed_frame=pd.DataFrame([x for x in per_seed if x["task"]==task]); row={"task":task,"seeds":len(args.seeds),**metrics(ensemble,args.threshold)}
        for col in ("auroc","pr_auc","sensitivity","specificity","balanced_accuracy"): row[f"seed_mean_{col}"]=float(seed_frame[col].mean()); row[f"seed_std_{col}"]=float(seed_frame[col].std(ddof=1))
        final.append(row)
    pd.DataFrame(per_seed).to_csv(args.output_dir/"per_seed_oof_metrics.csv",index=False); summary=pd.DataFrame(final); summary.to_csv(args.output_dir/"repeated_seed_ensemble_summary.csv",index=False); print(summary.to_string(index=False)); return 0
if __name__=="__main__": raise SystemExit(main())
