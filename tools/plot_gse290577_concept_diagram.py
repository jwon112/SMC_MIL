"""Render a presentation-ready concept diagram for the GSE290577 cohort."""

from __future__ import annotations

from pathlib import Path
import argparse

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch


def box(ax, x, y, width, height, text, color, *, fontsize=13, text_color="#17212b"):
    patch = FancyBboxPatch(
        (x, y), width, height,
        boxstyle="round,pad=0.018,rounding_size=0.025",
        facecolor=color, edgecolor="#53616c", linewidth=1.25,
    )
    ax.add_patch(patch)
    ax.text(x + width / 2, y + height / 2, text, ha="center", va="center", fontsize=fontsize, color=text_color, fontweight="bold")
    return patch


def arrow(ax, start, end, *, color="#53616c"):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=16, linewidth=1.5, color=color))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    plt.rcParams.update({"font.family": "Malgun Gothic", "axes.unicode_minus": False})
    fig, ax = plt.subplots(figsize=(16, 9), facecolor="white")
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 9)
    ax.axis("off")

    ax.text(8, 8.55, "GSE290577 공개 외부 검증 코호트", ha="center", va="center", fontsize=25, fontweight="bold")
    ax.text(8, 8.12, "내부 SMC gold cohort 학습에는 포함하지 않고, 독립 external test set으로만 사용", ha="center", va="center", fontsize=13, color="#45525c")
    box(ax, 5.3, 7.18, 5.4, 0.55, "공개 심장 이식 생검 이미지 + 공간전사체 연계 코호트", "#e8eef2", fontsize=14)

    # WSI branch
    box(ax, 0.75, 5.72, 6.45, 0.78, "WSI biopsy cohort\n15개 biopsy event, 총 60 WSI", "#c8e9e7", fontsize=16)
    ax.text(3.98, 5.40, "비거부 5 / ACR 5 / AMR 5", ha="center", fontsize=12, color="#45525c")
    box(ax, 1.05, 4.25, 2.6, 0.82, "H&E WSI\n15장", "#edf7f6", fontsize=14)
    box(ax, 4.30, 4.25, 2.6, 0.82, "IHC WSI\n45장: CD3, CD8, CD68", "#edf7f6", fontsize=13)
    arrow(ax, (3.2, 5.72), (2.4, 5.07), color="#278f8c")
    arrow(ax, (4.75, 5.72), (5.6, 5.07), color="#278f8c")
    box(ax, 1.05, 2.45, 5.85, 0.96, "WSI-level external evaluation\nH&E 단독 또는 H&E + IHC를 biopsy event 단위로 집계", "#d9f0ed", fontsize=13)
    arrow(ax, (2.35, 4.25), (3.05, 3.41), color="#278f8c")
    arrow(ax, (5.6, 4.25), (4.95, 3.41), color="#278f8c")

    # Core branch
    box(ax, 8.8, 5.72, 6.45, 0.78, "Xenium spatial core cohort\n62명, 195개 H&E tissue core ROI", "#f7d9bd", fontsize=16)
    ax.text(12.02, 5.40, "2개 OME-TIFF whole-tissue image에서 ROI 추출", ha="center", fontsize=12, color="#45525c")
    box(ax, 9.12, 4.25, 2.65, 0.82, "대형 H&E 조직 이미지\n2장", "#fff3e8", fontsize=14)
    box(ax, 12.38, 4.25, 2.55, 0.82, "공간좌표 기반\ncore ROI 195개", "#fff3e8", fontsize=14)
    arrow(ax, (11.32, 4.66), (12.38, 4.66), color="#c4712f")
    box(ax, 9.12, 2.45, 5.82, 0.96, "Core-level external evaluation\ncore는 환자 내 상관성이 있으므로 환자 단위 bootstrap으로 불확실성 평가", "#fbe8d6", fontsize=13)
    arrow(ax, (13.65, 4.25), (12.0, 3.41), color="#c4712f")

    # Shared processing and outputs
    box(ax, 3.6, 1.10, 8.8, 0.78, "공통 처리: 조직영역 mask → 물리 해상도 기반 patch (0.25/0.5/1/2 µm/px) → UNI-v2 feature", "#e9e5f5", fontsize=14)
    arrow(ax, (3.98, 2.45), (6.15, 1.88), color="#53616c")
    arrow(ax, (12.0, 2.45), (9.85, 1.88), color="#53616c")
    box(ax, 3.6, 0.22, 8.8, 0.52, "보고 원칙: WSI와 core 성능을 합산하지 않고 분리 보고 (서로 다른 표본 단위 및 독립성)", "#f1f3f5", fontsize=12)

    fig.tight_layout(pad=0)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=220, facecolor="white")


if __name__ == "__main__":
    main()
