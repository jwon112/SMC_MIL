from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Tuple


def _read_last_row(csv_path: Path) -> Dict[str, str]:
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"Empty CSV: {csv_path}")
    return rows[-1]


def _collect_result_csvs(runs_dir: Path, exp_dirs: List[str]) -> List[Path]:
    if exp_dirs:
        out: List[Path] = []
        for d in exp_dirs:
            p = Path(d)
            if not p.is_absolute():
                p = runs_dir / p
            out.append(p / "results" / "results.csv")
        return out

    # default: load all runs/*/results/results.csv
    out = []
    if runs_dir.exists():
        for d in sorted(runs_dir.iterdir()):
            if d.is_dir():
                p = d / "results" / "results.csv"
                if p.exists():
                    out.append(p)
    return out


def _to_float(x: str, default: float = float("nan")) -> float:
    try:
        if x is None:
            return default
        s = str(x).strip()
        if s == "" or s.lower() == "none":
            return default
        return float(s)
    except Exception:
        return default


def _plot(rows: List[Tuple[str, Dict[str, str]]], out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = [n for n, _ in rows]
    best_val_dice = [_to_float(r.get("best_val_dice", "")) for _, r in rows]
    test_dice = [_to_float(r.get("test_dice", "")) for _, r in rows]
    latency = [_to_float(r.get("latency_ms", "")) for _, r in rows]
    params_m = [_to_float(r.get("params", "")) / 1e6 for _, r in rows]
    flops_g = [_to_float(r.get("flops", "")) / 1e9 for _, r in rows]

    fig = plt.figure(figsize=(12, 8), dpi=160)
    gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.25)

    ax = fig.add_subplot(gs[0, 0])
    ax.bar(names, best_val_dice, label="best val dice")
    ax.bar(names, test_dice, alpha=0.7, label="test dice")
    ax.set_title("Dice")
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, axis="y", alpha=0.3)
    ax.tick_params(axis="x", rotation=30, labelsize=8)
    ax.legend()

    ax = fig.add_subplot(gs[0, 1])
    ax.bar(names, latency)
    ax.set_title("Latency (ms) - batch=1 dummy")
    ax.grid(True, axis="y", alpha=0.3)
    ax.tick_params(axis="x", rotation=30, labelsize=8)

    ax = fig.add_subplot(gs[1, 0])
    ax.bar(names, params_m)
    ax.set_title("Params (M)")
    ax.grid(True, axis="y", alpha=0.3)
    ax.tick_params(axis="x", rotation=30, labelsize=8)

    ax = fig.add_subplot(gs[1, 1])
    ax.bar(names, flops_g)
    ax.set_title("FLOPs (G) - single forward")
    ax.grid(True, axis="y", alpha=0.3)
    ax.tick_params(axis="x", rotation=30, labelsize=8)

    fig.suptitle("Experiment comparison")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--runs_dir", type=str, default="./runs", help="Directory containing experiment folders")
    p.add_argument(
        "--exp_dirs",
        type=str,
        nargs="*",
        default=[],
        help="Experiment directories to compare (absolute or relative to runs_dir). If omitted, loads all.",
    )
    p.add_argument("--out", type=str, default="./compare/compare.png")
    p.add_argument("--out_csv", type=str, default="./compare/compare.csv")
    args = p.parse_args()

    runs_dir = Path(args.runs_dir)
    csv_paths = _collect_result_csvs(runs_dir, args.exp_dirs)
    if not csv_paths:
        raise SystemExit(f"No results.csv found under {runs_dir}")

    rows: List[Tuple[str, Dict[str, str]]] = []
    for pth in csv_paths:
        row = _read_last_row(pth)
        # name = experiment folder name (runs/<name>/results/results.csv)
        name = pth.parents[1].name
        rows.append((name, row))

    # write merged sheet
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({k for _, r in rows for k in r.keys()})
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["exp_name"] + keys)
        w.writeheader()
        for name, r in rows:
            rr = {"exp_name": name}
            rr.update(r)
            w.writerow(rr)

    _plot(rows, Path(args.out))
    print(f"wrote {out_csv}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

