"""Render a compact external-AUROC comparison for the 5-seed ensemble slide."""

from __future__ import annotations

from pathlib import Path
import argparse

import matplotlib.pyplot as plt
import numpy as np


ROWS = [
    ("ACR", "WSI H&E", 15, 0.896, 0.960),
    ("ACR", "WSI IHC", 15, 0.745, 0.858),
    ("ACR", "Core", 189, 0.766, 0.786),
    ("AMR", "Core", 187, 0.566, 0.583),
    ("Significant rejection", "Core", 186, 0.740, 0.758),
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    plt.rcParams.update({"font.family": "Malgun Gothic", "axes.unicode_minus": False})
    fig = plt.figure(figsize=(16, 8.5), facecolor="white")
    table_ax = fig.add_axes((0.06, 0.20, 0.62, 0.64))
    delta_ax = fig.add_axes((0.73, 0.24, 0.22, 0.53))
    table_ax.axis("off")

    fig.text(0.06, 0.92, "5-seed ensemble: 외부 검증 AUROC의 일관된 개선", fontsize=25, fontweight="bold")
    fig.text(0.06, 0.875, "기준 run 대비 ensemble 예측 확률 평균. 다섯 비교 모두 AUROC가 상승함.", fontsize=13, color="#53616c")

    headers = ["Task", "외부 평가 단위", "n", "기준 run", "5-seed ensemble", "ΔAUROC"]
    x = np.array([0.00, 0.22, 0.48, 0.57, 0.73, 0.94])
    for xpos, header in zip(x, headers):
        table_ax.text(xpos, 0.97, header, transform=table_ax.transAxes, ha="left", va="center", fontsize=13, fontweight="bold", color="#1b2933")
    table_ax.plot([0, 1], [0.91, 0.91], transform=table_ax.transAxes, color="#5e6c76", linewidth=1.4)

    for index, (task, unit, n, baseline, ensemble) in enumerate(ROWS):
        y = 0.82 - index * 0.16
        background = "#f4f7f8" if index % 2 == 0 else "white"
        table_ax.add_patch(plt.Rectangle((0, y - 0.075), 1, 0.125, transform=table_ax.transAxes, facecolor=background, edgecolor="none"))
        delta = ensemble - baseline
        cells = [task, unit, f"{n}", f"{baseline:.3f}", f"{ensemble:.3f}", f"+{delta:.3f}"]
        colors = ["#1b2933", "#354550", "#354550", "#354550", "#007c85", "#007c85"]
        weights = ["normal", "normal", "normal", "normal", "bold", "bold"]
        for xpos, value, color, weight in zip(x, cells, colors, weights):
            table_ax.text(xpos, y, value, transform=table_ax.transAxes, ha="left", va="center", fontsize=14, color=color, fontweight=weight)
    table_ax.plot([0, 1], [0.095, 0.095], transform=table_ax.transAxes, color="#d4dce1", linewidth=1.0)

    deltas = np.array([ensemble - baseline for _, _, _, baseline, ensemble in ROWS])
    labels = ["ACR\nH&E", "ACR\nIHC", "ACR\ncore", "AMR\ncore", "Significant\ncore"]
    bars = delta_ax.barh(np.arange(len(deltas)), deltas, color="#168a98", height=0.58)
    delta_ax.set_yticks(np.arange(len(deltas)), labels, fontsize=11)
    delta_ax.invert_yaxis()
    delta_ax.set_xlim(0, 0.13)
    delta_ax.set_xlabel("ΔAUROC", fontsize=12)
    delta_ax.set_title("개선 폭", fontsize=15, fontweight="bold", pad=12)
    delta_ax.xaxis.grid(True, color="#dce3e7")
    delta_ax.set_axisbelow(True)
    delta_ax.spines[["top", "right", "left"]].set_visible(False)
    for bar, value in zip(bars, deltas):
        delta_ax.text(value + 0.003, bar.get_y() + bar.get_height() / 2, f"+{value:.3f}", va="center", fontsize=11, fontweight="bold", color="#00717c")

    fig.text(0.06, 0.085, "해석 시 유의: WSI는 n=15로 작고, core는 195개 ROI가 62명에서 유래하므로 core 단위를 독립 환자로 간주하지 않음.", fontsize=11.5, color="#53616c")
    fig.text(0.06, 0.052, "AUROC는 threshold-independent 지표이며, external set은 checkpoint·threshold 선택에 사용하지 않음.", fontsize=11.5, color="#53616c")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=220, facecolor="white")


if __name__ == "__main__":
    main()
