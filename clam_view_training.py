"""CLAM-SB training controls for the exploratory UNI2 view-bank experiment.

The same validation fold controls LR, stopping and checkpoint selection; metrics
are internal exploratory validation, not an independent held-out assessment.
Model architecture, CE + instance loss and slide sampling follow the existing
CLAM experiment. This module owns the loop so all control parameters take effect.
"""
from __future__ import annotations

import csv
import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class LossMonitor:
    patience: int = 15
    min_epochs: int = 10
    min_delta: float = 1e-4
    best_loss: float = math.inf
    best_epoch: int = 0
    reference_loss: float = math.inf
    bad_epochs: int = 0

    def update(self, epoch: int, loss: float) -> tuple[bool, bool]:
        if not math.isfinite(loss):
            raise ValueError(f"Non-finite validation loss at epoch {epoch}")
        # Save every strictly better model, even if below the patience delta.
        best = loss < self.best_loss
        if best:
            self.best_loss, self.best_epoch = loss, epoch
        if loss < self.reference_loss - self.min_delta:
            self.reference_loss, self.bad_epochs = loss, 0
        else:
            self.bad_epochs += 1
        return best, epoch >= self.min_epochs and self.bad_epochs >= self.patience


def training_config(args) -> dict:
    return {
        "training_protocol": "best_val_v2", "model": "clam_sb", "embed_dim": 1536,
        "seed": args.seed, "sampler_seed": args.seed + args.fold,
        "view_seed": args.view_seed + args.fold,
        "sampler": args.sampler, "optimizer": "Adam", "lr": args.lr,
        "weight_decay": args.reg, "dropout": args.drop_out,
        "model_size": args.model_size, "bag_weight": args.bag_weight,
        "instance_clustering": not args.no_inst_cluster, "B": args.B,
        "max_epochs": args.max_epochs, "early_stopping": args.early_stopping,
        "early_stop_patience": args.early_stop_patience,
        "early_stop_min_epochs": args.early_stop_min_epoch,
        "early_stop_min_delta": args.early_stop_min_delta,
        "checkpoint_monitor": "val_loss", "checkpoint_mode": "min",
        "grad_clip_norm": args.grad_clip,
        "lr_scheduler": "ReduceLROnPlateau", "lr_factor": args.lr_factor,
        "lr_patience": args.lr_patience, "min_lr": args.min_lr,
        "precision": "float32", "epoch_index_base": 1,
        "validation_used_for_checkpoint_lr_and_stopping": True,
        "train_eval": "all original train bags once in eval mode each epoch",
    }


def atomic_json(path: Path, record: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def save_checkpoint(path: Path, model, epoch: int, val_loss: float, config: dict) -> None:
    import torch

    temporary = path.with_suffix(".pt.tmp")
    torch.save({"state_dict": model.state_dict(), "epoch": epoch,
                "val_loss": val_loss, "training_config": config}, temporary)
    os.replace(temporary, path)


def load_checkpoint(path: Path, model, device) -> dict:
    import torch

    try:
        checkpoint = torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    return checkpoint


def binary_metrics(labels, scores, predictions) -> dict:
    from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

    y = np.asarray(labels, dtype=int)
    p = np.asarray(scores, dtype=float)
    pred = np.asarray(predictions, dtype=int)
    if len(y) == 0 or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("Invalid predictions")
    both = len(np.unique(y)) == 2
    return {
        "error": float(np.mean(pred != y)), "acc": float(np.mean(pred == y)),
        "auc": float(roc_auc_score(y, p)) if both else None,
        "ap": float(average_precision_score(y, p)) if both else None,
        "brier": float(brier_score_loss(y, p)),
        "class0_acc": float(np.mean(pred[y == 0] == 0)) if np.any(y == 0) else None,
        "class1_acc": float(np.mean(pred[y == 1] == 1)) if np.any(y == 1) else None,
    }


def evaluate_original(model, slide_data, baseline_dir: Path, device, feature_loader):
    import torch

    model.eval()
    losses, labels, scores, predictions = [], [], [], []
    outputs = {}
    with torch.inference_mode():
        for row in slide_data.to_dict("records"):
            sid, label = str(row["slide_id"]), int(row["label"])
            bag = feature_loader(baseline_dir / "pt_files" / f"{sid}.pt").to(device)
            target = torch.tensor([label], dtype=torch.long, device=device)
            logits, probability, prediction, _, _ = model(bag)
            if not torch.isfinite(logits).all():
                raise ValueError(f"Non-finite validation logits: {sid}")
            loss = torch.nn.functional.cross_entropy(logits, target)
            score = float(probability[0, 1].item())
            losses.append(float(loss.item()))
            labels.append(label)
            scores.append(score)
            predictions.append(int(prediction.reshape(-1)[0].item()))
            outputs[sid] = {"prob": probability.detach().cpu().numpy(), "label": label}
    metrics = {"loss": float(np.mean(losses)), **binary_metrics(labels, scores, predictions)}
    if {"case_id", "event_id"} <= set(slide_data.columns):
        events = slide_data[["case_id", "event_id"]].copy().reset_index(drop=True)
        events["label"], events["score"] = labels, scores
        if events.groupby(["case_id", "event_id"])["label"].nunique().gt(1).any():
            raise ValueError("Conflicting labels within an event")
        events = events.groupby(["case_id", "event_id"], as_index=False).agg(
            label=("label", "first"), score=("score", "mean"))
        event_metrics = binary_metrics(events.label, events.score, (events.score > 0.5).astype(int))
        metrics.update({"event_ap": event_metrics["ap"], "event_auc": event_metrics["auc"],
                        "event_brier": event_metrics["brier"]})
    return outputs, metrics


def draw_indices(labels: list[int], sampler: str, generator) -> list[int]:
    import torch

    if sampler == "uniform":
        return torch.randperm(len(labels), generator=generator).tolist()
    counts = np.bincount(labels, minlength=2)
    if (counts == 0).any():
        raise ValueError("Weighted sampling needs both classes")
    weights = torch.tensor([1.0 / counts[label] for label in labels], dtype=torch.double)
    return torch.multinomial(weights, len(labels), replacement=True, generator=generator).tolist()


def plot_history(rows: list[dict], destination: Path, best_epoch: int) -> bool:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    frame = pd.DataFrame(rows)
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    for key, axis in zip(("loss", "error", "acc"), axes.flat):
        for prefix, title in (("train", "Train: sampled, dropout on"),
                              ("train_original", "Train: original, eval mode"),
                              ("val", "Validation: original")):
            axis.plot(frame.epoch, frame[f"{prefix}_{key}"], label=title)
        axis.set(xlabel="Epoch", ylabel=key)
        axis.axvline(best_epoch, color="grey", linestyle="--", linewidth=1)
        axis.legend(fontsize=7)
    axes[1, 1].plot(frame.epoch, frame.val_auc, label="Val slide AUROC")
    axes[1, 1].plot(frame.epoch, frame.val_event_ap, label="Val event AP")
    axes[1, 1].set(xlabel="Epoch", ylabel="Validation metric", ylim=(0, 1))
    axes[1, 1].legend(fontsize=8)
    temporary = destination.with_suffix(".tmp.png")
    fig.savefig(temporary, dpi=140)
    plt.close(fig)
    os.replace(temporary, destination)
    return True


def fit_best_checkpoint(train_split, val_split, args, *, model=None, feature_loader=None):
    import torch
    from feature_view_bank import torch_feature_loader

    feature_loader = feature_loader or torch_feature_loader
    device = torch.device(getattr(args, "device", "cuda" if torch.cuda.is_available() else "cpu"))
    if model is None:
        from models.model_clam import CLAM_SB
        model = CLAM_SB(size_arg=args.model_size, dropout=args.drop_out, k_sample=args.B,
                        n_classes=2, instance_loss_fn=torch.nn.CrossEntropyLoss(),
                        subtyping=False, embed_dim=1536)
    model = model.to(device)
    config = training_config(args)
    output = Path(args.results_dir)
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(output / "training_config.json", config)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.reg)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=args.lr_factor, patience=args.lr_patience,
        threshold=args.early_stop_min_delta, threshold_mode="abs", min_lr=args.min_lr)
    monitor = LossMonitor(args.early_stop_patience, args.early_stop_min_epoch,
                          args.early_stop_min_delta)
    sampler_rng = torch.Generator().manual_seed(args.seed + args.fold)
    train_labels = [int(label) for label in train_split.slide_data.label]
    history = []
    steps = 0
    stopped = False
    started = time.perf_counter()
    curves_available = False
    best_path = output / "best_checkpoint.pt"
    with (output / "epoch_metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = None
        for epoch in range(1, args.max_epochs + 1):
            epoch_started = time.perf_counter()
            model.train()
            lr_used = float(optimizer.param_groups[0]["lr"])
            sums = dict(loss=0., total_loss=0., instance_loss=0., grad_norm=0., clipped=0.)
            ys, ps, predictions = [], [], []
            for index in draw_indices(train_labels, args.sampler, sampler_rng):
                bag, label = train_split[index]
                bag = bag.to(device)
                target = torch.tensor([int(label)], device=device, dtype=torch.long)
                if hasattr(model, "k_sample"):
                    model.k_sample = min(args.B, len(bag))
                optimizer.zero_grad(set_to_none=True)
                logits, prob, pred, _, detail = model(
                    bag, label=target, instance_eval=not args.no_inst_cluster)
                bag_loss = torch.nn.functional.cross_entropy(logits, target)
                instance_loss = (detail["instance_loss"] if not args.no_inst_cluster
                                 else torch.zeros((), device=device))
                total = (args.bag_weight * bag_loss + (1 - args.bag_weight) * instance_loss
                         if not args.no_inst_cluster else bag_loss)
                if not torch.isfinite(total):
                    raise ValueError(f"Non-finite loss at epoch={epoch}, train_index={index}")
                total.backward()
                norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), args.grad_clip, error_if_nonfinite=True)
                optimizer.step()
                steps += 1
                sums["loss"] += float(bag_loss.item())
                sums["total_loss"] += float(total.item())
                sums["instance_loss"] += float(instance_loss.item())
                sums["grad_norm"] += float(norm)
                sums["clipped"] += float(norm > args.grad_clip)
                ys.append(int(label))
                ps.append(float(prob[0, 1].detach().item()))
                predictions.append(int(pred.reshape(-1)[0].item()))
            n = len(ys)
            train_metrics = {key: value / n for key, value in sums.items()}
            train_metrics.update(binary_metrics(ys, ps, predictions))
            _, original_metrics = evaluate_original(
                model, train_split.slide_data, args.baseline_dir, device, feature_loader)
            _, val_metrics = evaluate_original(
                model, val_split.slide_data, args.baseline_dir, device, feature_loader)
            is_best, should_stop = monitor.update(epoch, val_metrics["loss"])
            if is_best:
                save_checkpoint(best_path, model, epoch, val_metrics["loss"], config)
            save_checkpoint(output / "last_checkpoint.pt", model, epoch, val_metrics["loss"], config)
            scheduler.step(val_metrics["loss"])
            row = {"epoch": epoch, "lr": lr_used,
                   "lr_next": float(optimizer.param_groups[0]["lr"]),
                   **{f"train_{key}": value for key, value in train_metrics.items()},
                   **{f"train_original_{key}": value for key, value in original_metrics.items()},
                   **{f"val_{key}": value for key, value in val_metrics.items()},
                   "best_epoch": monitor.best_epoch, "is_best": is_best,
                   "bad_epochs": monitor.bad_epochs, "optimizer_steps": steps,
                   "epoch_seconds": time.perf_counter() - epoch_started}
            if writer is None:
                writer = csv.DictWriter(stream, fieldnames=list(row))
                writer.writeheader()
            writer.writerow(row)
            stream.flush()
            history.append(row)
            stopped = bool(args.early_stopping and should_stop)
            atomic_json(output / "training_status.json", {
                "epoch": epoch, "best_epoch": monitor.best_epoch,
                "best_val_loss": monitor.best_loss, "bad_epochs": monitor.bad_epochs,
                "lr_next": row["lr_next"], "stopped_early": stopped, "complete": False})
            print(f"Epoch {epoch}/{args.max_epochs}: train_error={row['train_error']:.4f}, "
                  f"train_original_error={row['train_original_error']:.4f}, "
                  f"val_error={row['val_error']:.4f}, val_loss={row['val_loss']:.4f}, "
                  f"val_auc={row['val_auc']:.4f}, lr={lr_used:.2e}, "
                  f"best_epoch={monitor.best_epoch}, patience={monitor.bad_epochs}/{monitor.patience}",
                  flush=True)
            if epoch == 1 or epoch % 5 == 0 or stopped or epoch == args.max_epochs:
                curves_available = plot_history(history, output / "learning_curves.png", monitor.best_epoch)
            if stopped:
                print(f"Early stopping at epoch {epoch}; restoring best epoch {monitor.best_epoch}", flush=True)
                break
    checkpoint = load_checkpoint(best_path, model, device)
    results, metrics = evaluate_original(model, val_split.slide_data, args.baseline_dir, device, feature_loader)
    if not math.isclose(metrics["loss"], checkpoint["val_loss"], rel_tol=1e-5, abs_tol=1e-6):
        raise RuntimeError("Reloaded best checkpoint does not reproduce its validation loss")
    report = {
        "training_protocol": "best_val_v2", "training_config": config,
        "checkpoint_monitor": "val_loss", "checkpoint_mode": "min",
        "best_epoch": monitor.best_epoch, "best_val_loss": monitor.best_loss,
        "epochs_ran": len(history), "stopped_early": stopped,
        "optimizer_steps": steps, "training_seconds": time.perf_counter() - started,
        "selected_checkpoint": str(best_path), "learning_curves_available": curves_available,
        "selection_uses_reported_validation_fold": True,
        "epoch_index_base": 1,
    }
    atomic_json(output / "training_status.json", {**report, "complete": True})
    return results, metrics["auc"], report
