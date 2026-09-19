#!/usr/bin/env python3
"""Train hierarchical stain-aware event MIL on patient-grouped gold folds."""
from __future__ import annotations
import argparse, json, random
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from models.stain_aware_event_mil import StainAwareEventMIL
from utils.event_mil import EventFeatureDataset, collate_event, load_event_tables, verify_feature_bags

def arguments() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("event-csv", "event-slides-csv", "split-dir", "feature-dir", "results-dir"): p.add_argument(f"--{name}", type=Path, required=True)
    p.add_argument("--folds", type=int, default=5); p.add_argument("--seed", type=int, default=1); p.add_argument("--input-dim", type=int, default=1536); p.add_argument("--hidden-dim", type=int, default=128); p.add_argument("--dropout", type=float, default=.25)
    p.add_argument("--max-patches-per-slide", type=int, default=2048); p.add_argument("--max-epochs", type=int, default=50); p.add_argument("--patience", type=int, default=10); p.add_argument("--min-epochs", type=int, default=10); p.add_argument("--lr", type=float, default=2e-4); p.add_argument("--weight-decay", type=float, default=1e-5); p.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    return p.parse_args()
def seed_all(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[float, pd.DataFrame]:
    model.eval(); loss_fn = nn.CrossEntropyLoss(); losses=[]; rows=[]
    with torch.inference_mode():
        for batch in loader:
            logits, _ = model([(stain, features.to(device)) for stain, features in batch["slides"]]); label = torch.tensor([batch["label"]], device=device); losses.append(float(loss_fn(logits, label).cpu()))
            rows.append({"event_id": batch["event_id"], "case_id": batch["case_id"], "label": batch["label"], "probability": float(torch.softmax(logits, 1)[0, 1].cpu())})
    return float(np.mean(losses)), pd.DataFrame(rows)
def score(frame: pd.DataFrame) -> dict[str, float | int]:
    y, p = frame.label.to_numpy(int), frame.probability.to_numpy(float); tn, fp, fn, tp = confusion_matrix(y, p >= .5, labels=[0, 1]).ravel()
    return {"events": len(frame), "positive_events": int(y.sum()), "auroc": float(roc_auc_score(y, p)), "pr_auc": float(average_precision_score(y, p)), "sensitivity": float(tp/(tp+fn)) if tp+fn else float("nan"), "specificity": float(tn/(tn+fp)) if tn+fp else float("nan")}
def main() -> int:
    args = arguments(); seed_all(args.seed); device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else args.device if args.device != "auto" else "cpu")
    events, slides = load_event_tables(args.event_csv, args.event_slides_csv); verify_feature_bags(slides, args.feature_dir); args.results_dir.mkdir(parents=True, exist_ok=True); (args.results_dir / "run_config.json").write_text(json.dumps(vars(args), default=str, indent=2), encoding="utf-8")
    all_oof=[]; records=[]
    for fold in range(args.folds):
        split = pd.read_csv(args.split_dir / f"splits_{fold}.csv", dtype=str); train = events.loc[events.event_id.isin(set(split.train.dropna()))]; val = events.loc[events.event_id.isin(set(split.val.dropna()))]
        train_ds = EventFeatureDataset(train, slides, args.feature_dir, args.max_patches_per_slide, True); val_ds = EventFeatureDataset(val, slides, args.feature_dir, args.max_patches_per_slide, False)
        counts=train.label.value_counts(); weights=train.label.map({label: 1.0/count for label, count in counts.items()}).to_numpy(); train_loader=DataLoader(train_ds, batch_size=1, sampler=WeightedRandomSampler(weights, len(weights), replacement=True), collate_fn=collate_event); val_loader=DataLoader(val_ds, batch_size=1, shuffle=False, collate_fn=collate_event)
        model=StainAwareEventMIL(args.input_dim, args.hidden_dim, args.dropout).to(device); optimizer=torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay); loss_fn=nn.CrossEntropyLoss(); best=float("inf"); stale=0; ckpt=args.results_dir/f"s_{fold}_checkpoint.pt"
        for epoch in range(args.max_epochs):
            model.train(); losses=[]
            for batch in train_loader:
                optimizer.zero_grad(); logits,_=model([(stain, feature.to(device)) for stain, feature in batch["slides"]]); loss=loss_fn(logits, torch.tensor([batch["label"]], device=device)); loss.backward(); optimizer.step(); losses.append(float(loss.detach().cpu()))
            val_loss,_=evaluate(model,val_loader,device); print(f"fold={fold} epoch={epoch+1} train_loss={np.mean(losses):.4f} val_loss={val_loss:.4f}")
            if val_loss < best: best=val_loss; stale=0; torch.save({"state_dict":model.state_dict(),"model_config":{"input_dim":args.input_dim,"hidden_dim":args.hidden_dim,"dropout":args.dropout}},ckpt)
            else: stale += 1
            if epoch+1 >= args.min_epochs and stale >= args.patience: break
        model.load_state_dict(torch.load(ckpt,map_location=device,weights_only=True)["state_dict"]); _,oof=evaluate(model,val_loader,device); oof["fold"]=fold; oof.to_csv(args.results_dir/f"fold_{fold}_oof_predictions.csv",index=False); row={"fold":fold,"best_val_loss":best,**score(oof)}; records.append(row); all_oof.append(oof); print("[OK] "+str(row))
    oof=pd.concat(all_oof,ignore_index=True); oof.to_csv(args.results_dir/"oof_predictions.csv",index=False); pd.DataFrame(records).to_csv(args.results_dir/"fold_summary.csv",index=False); print("[OOF] "+str(score(oof))); return 0
if __name__ == "__main__": raise SystemExit(main())
