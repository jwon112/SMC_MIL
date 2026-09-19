#!/usr/bin/env python3
"""Evaluate a stain-aware event-MIL fold ensemble on external biopsy events."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, average_precision_score, confusion_matrix, roc_auc_score
from models.stain_aware_event_mil import StainAwareEventMIL
from utils.event_mil import normalize_stain_group

TASK_LABELS={"acr_high":"acr_high_label","amr_positive":"amr_positive_label","significant_rejection":"significant_rejection_label"}
def arguments() -> argparse.Namespace:
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--checkpoint-dir",type=Path,required=True); p.add_argument("--cohort",nargs=3,action="append",metavar=("NAME","MANIFEST","FEATURE_DIR"),required=True); p.add_argument("--task",choices=TASK_LABELS,required=True); p.add_argument("--output-dir",type=Path,required=True); p.add_argument("--threshold",type=float,default=.5); p.add_argument("--max-patches-per-slide",type=int,default=2048); p.add_argument("--device",choices=("auto","cuda","cpu"),default="auto"); return p.parse_args()
def event_column(frame: pd.DataFrame, cohort: str) -> str:
    if cohort.lower().startswith("core"): return "slide_id"
    for name in ("biopsy_id","event_id","case_id","patient_id"):
        if name in frame: return name
    raise ValueError("Manifest requires biopsy_id, event_id, case_id, or patient_id")
def score(y: np.ndarray, p: np.ndarray, threshold: float) -> dict[str,float|int]:
    pred=p>=threshold; tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel(); sens=float(tp/(tp+fn)) if tp+fn else float("nan"); spec=float(tn/(tn+fp)) if tn+fp else float("nan")
    return {"events":len(y),"positive_events":int(y.sum()),"auroc":float(roc_auc_score(y,p)) if len(np.unique(y))==2 else float("nan"),"pr_auc":float(average_precision_score(y,p)) if y.sum() else float("nan"),"accuracy":float(accuracy_score(y,pred)),"sensitivity":sens,"specificity":spec,"balanced_accuracy":float((sens+spec)/2),"threshold":threshold,"tp":int(tp),"tn":int(tn),"fp":int(fp),"fn":int(fn)}
def main() -> int:
    args=arguments(); device=torch.device("cuda" if args.device=="auto" and torch.cuda.is_available() else args.device if args.device!="auto" else "cpu"); checkpoints=sorted(args.checkpoint_dir.glob("s_*_checkpoint.pt"))
    if not checkpoints: raise FileNotFoundError(f"No s_*_checkpoint.pt in {args.checkpoint_dir}")
    models=[]
    for path in checkpoints:
        payload=torch.load(path,map_location=device,weights_only=True); model=StainAwareEventMIL(**payload["model_config"]).to(device); model.load_state_dict(payload["state_dict"]); models.append(model.eval())
    print(f"Loaded {len(models)} fold checkpoints"); args.output_dir.mkdir(parents=True,exist_ok=True); summaries=[]; label_col=TASK_LABELS[args.task]
    for cohort, manifest_path, feature_path in args.cohort:
        frame=pd.read_csv(manifest_path,dtype=str).fillna(""); key=event_column(frame, cohort); frame=frame.loc[frame[label_col].astype(str).str.strip().ne("")].copy(); frame["_stain"]=frame.stain_group.map(normalize_stain_group); frame=frame.loc[frame._stain.notna()].copy()
        if frame.empty: raise ValueError(f"No labeled known-stain rows in {manifest_path}")
        if (frame.groupby(key)[label_col].nunique()>1).any(): raise ValueError(f"{cohort}: inconsistent labels within event")
        rows=[]
        for event_id, group in frame.groupby(key,sort=True):
            slides=[]
            for _,row in group.iterrows():
                bag=Path(feature_path)/"pt_files"/f"{row['slide_id']}.pt"
                if not bag.is_file(): raise FileNotFoundError(f"Missing feature bag: {bag}")
                features=torch.load(bag,map_location=device,weights_only=True).float()
                if len(features) > args.max_patches_per_slide:
                    indices=torch.linspace(0,len(features)-1,args.max_patches_per_slide).long(); features=features[indices]
                slides.append((row["_stain"],features.to(device)))
            with torch.inference_mode(): probs=[float(torch.softmax(model(slides)[0],1)[0,1].cpu()) for model in models]
            rows.append({"event_id":str(event_id),"label":int(float(group.iloc[0][label_col])),"probability":float(np.mean(probs)),"slides":len(slides),"stain_groups":"+".join(sorted(group._stain.unique())),**{f"probability_fold_{i}":p for i,p in enumerate(probs)}})
        result=pd.DataFrame(rows); result["prediction"]=(result.probability>=args.threshold).astype(int); result.to_csv(args.output_dir/f"{cohort}_event_predictions.csv",index=False); metrics={"cohort":cohort,"task":args.task,**score(result.label.to_numpy(int),result.probability.to_numpy(float),args.threshold)}; summaries.append(metrics); print(f"[OK] {cohort}: n={metrics['events']}, positive={metrics['positive_events']}, AUROC={metrics['auroc']:.3f}, PR-AUC={metrics['pr_auc']:.3f}")
    pd.DataFrame(summaries).to_csv(args.output_dir/"external_summary.csv",index=False); return 0
if __name__=="__main__": raise SystemExit(main())
