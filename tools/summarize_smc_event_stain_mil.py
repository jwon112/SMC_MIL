#!/usr/bin/env python3
"""Summarize repeated event-MIL CV and create a five-seed OOF ensemble."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score, roc_curve

TASKS = ("acr_high", "amr_positive", "significant_rejection")
SCALES = {
    "40x": "l0_0p25mpp_40x",
    "20x": "l1_0p50mpp_20x",
    "10x": "l2_1p00mpp_10x",
    "5x": "l3_2p00mpp_5x",
}
def arguments() -> argparse.Namespace:
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--results-root",type=Path,required=True); p.add_argument("--output-dir",type=Path,required=True); p.add_argument("--seeds",nargs="+",type=int,default=[1,11,21,31,41]); p.add_argument("--threshold",type=float,default=.5); p.add_argument("--tasks",nargs="+",choices=TASKS,default=list(TASKS)); p.add_argument("--scales",nargs="+",choices=SCALES,default=["40x"]); p.add_argument("--modes",nargs="+",choices=("aware","aware_nomask","agnostic"),default=["aware"]); return p.parse_args()
def sensitivity_at_specificity(y: np.ndarray, p: np.ndarray, target: float) -> float:
    fpr, tpr, _ = roc_curve(y, p)
    eligible = tpr[(1.0 - fpr) >= target]
    return float(eligible.max()) if len(eligible) else float("nan")
def metrics(frame: pd.DataFrame, threshold: float) -> dict[str,float|int]:
    y=frame.label.to_numpy(int); p=frame.probability.to_numpy(float); tn,fp,fn,tp=confusion_matrix(y,p>=threshold,labels=[0,1]).ravel()
    prevalence=float(y.mean()); pr_auc=float(average_precision_score(y,p))
    return {"events":len(y),"positive_events":int(y.sum()),"prevalence":prevalence,"auroc":float(roc_auc_score(y,p)),"pr_auc":pr_auc,"pr_auc_lift":float(pr_auc/prevalence) if prevalence else float("nan"),"sensitivity":float(tp/(tp+fn)) if tp+fn else float("nan"),"specificity":float(tn/(tn+fp)) if tn+fp else float("nan"),"balanced_accuracy":float(((tp/(tp+fn))+(tn/(tn+fp)))/2) if tp+fn and tn+fp else float("nan"),"sensitivity_at_specificity_90":sensitivity_at_specificity(y,p,.90),"sensitivity_at_specificity_95":sensitivity_at_specificity(y,p,.95)}
def result_path(root: Path, task: str, scale: str, mode: str, seed: int) -> Path:
    suffix="" if mode == "aware" else f"_{mode}"
    current = root/f"{task}_{SCALES[scale]}{suffix}_seed{seed}"/"oof_predictions.csv"
    if current.is_file():
        return current
    # The original 40x stain-aware pilot predates the explicit magnification
    # suffix in result directory names. Keep those completed runs reusable.
    if scale == "40x" and mode == "aware":
        legacy = root/f"{task}_l0_0p25mpp_seed{seed}"/"oof_predictions.csv"
        if legacy.is_file():
            return legacy
    return current
def main() -> int:
    args=arguments(); args.output_dir.mkdir(parents=True,exist_ok=True); per_seed=[]; final=[]
    for task in args.tasks:
        for scale in args.scales:
            for mode in args.modes:
                frames=[]
                for seed in args.seeds:
                    path=result_path(args.results_root,task,scale,mode,seed)
                    if not path.is_file(): raise FileNotFoundError(path)
                    frame=pd.read_csv(path,dtype={"event_id":str,"case_id":str}); frame["seed"]=seed; frames.append(frame); per_seed.append({"task":task,"scale":scale,"mode":mode,"seed":seed,**metrics(frame,args.threshold)})
                combined=pd.concat(frames,ignore_index=True)
                if (combined.groupby("event_id").label.nunique()>1).any(): raise ValueError(f"{task}/{scale}/{mode}: inconsistent event labels across seeds")
                ensemble=combined.groupby("event_id",as_index=False).agg(case_id=("case_id","first"),label=("label","first"),probability=("probability","mean"),seed_predictions=("seed","nunique"))
                if not ensemble.seed_predictions.eq(len(args.seeds)).all(): raise ValueError(f"{task}/{scale}/{mode}: incomplete seed coverage")
                ensemble["prediction"]=(ensemble.probability>=args.threshold).astype(int); ensemble.to_csv(args.output_dir/f"{task}_{scale}_{mode}_seed_ensemble_oof.csv",index=False)
                seed_frame=pd.DataFrame([x for x in per_seed if x["task"]==task and x["scale"]==scale and x["mode"]==mode]); row={"task":task,"scale":scale,"mode":mode,"seeds":len(args.seeds),**metrics(ensemble,args.threshold)}
                for col in ("auroc","pr_auc","pr_auc_lift","sensitivity","specificity","balanced_accuracy","sensitivity_at_specificity_90","sensitivity_at_specificity_95"): row[f"seed_mean_{col}"]=float(seed_frame[col].mean()); row[f"seed_std_{col}"]=float(seed_frame[col].std(ddof=1))
                final.append(row)
    pd.DataFrame(per_seed).to_csv(args.output_dir/"per_seed_oof_metrics.csv",index=False); summary=pd.DataFrame(final); summary.to_csv(args.output_dir/"repeated_seed_ensemble_summary.csv",index=False); print(summary.to_string(index=False)); return 0
if __name__=="__main__": raise SystemExit(main())
