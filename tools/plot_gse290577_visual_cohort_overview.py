"""Create an image-led overview of the GSE290577 external cohort."""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


def add_card(ax, x, y, width, height, image: Image.Image, title: str, subtitle: str, accent: str) -> None:
    ax.add_patch(FancyBboxPatch((x, y), width, height, boxstyle="round,pad=0.02,rounding_size=0.025", facecolor="white", edgecolor="#60707c", linewidth=1.2))
    ax.imshow(image, extent=(x + 0.06, x + width - 0.06, y + 0.78, y + height - 0.08), aspect="auto", zorder=2)
    ax.add_patch(FancyBboxPatch((x + 0.06, y + 0.12), width - 0.12, 0.52, boxstyle="round,pad=0.012,rounding_size=0.02", facecolor=accent, edgecolor="none", zorder=3))
    ax.text(x + width / 2, y + 0.47, title, ha="center", va="center", fontsize=15, fontweight="bold", zorder=4)
    ax.text(x + width / 2, y + 0.25, subtitle, ha="center", va="center", fontsize=11.5, color="#27343e", zorder=4)


def box(ax, x, y, width, height, text, color, fontsize=14) -> None:
    ax.add_patch(FancyBboxPatch((x, y), width, height, boxstyle="round,pad=0.02,rounding_size=0.025", facecolor=color, edgecolor="#60707c", linewidth=1.2))
    ax.text(x + width / 2, y + height / 2, text, ha="center", va="center", fontsize=fontsize, fontweight="bold", color="#1d2a34")


def arrow(ax, start, end, color="#60707c") -> None:
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=16, linewidth=1.5, color=color))


def load_preview(path: Path, max_size: tuple[int, int]) -> Image.Image:
    Image.MAX_IMAGE_PIXELS = None
    with Image.open(path) as image:
        image.thumbnail(max_size)
        return image.convert("RGB").copy()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--he-wsi", type=Path, required=True)
    parser.add_argument("--ihc-wsi", type=Path, required=True)
    parser.add_argument("--spatial-image", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    plt.rcParams.update({"font.family": "Malgun Gothic", "axes.unicode_minus": False})
    he = load_preview(args.he_wsi, (1000, 560))
    ihc = load_preview(args.ihc_wsi, (1000, 560))
    spatial = load_preview(args.spatial_image, (600, 900))

    fig, ax = plt.subplots(figsize=(16, 9), facecolor="white")
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 9)
    ax.axis("off")

    ax.text(8, 8.52, "GSE290577: 공개 심장 이식 거부반응 외부 검증 코호트", ha="center", va="center", fontsize=23, fontweight="bold")
    ax.text(8, 8.08, "내부 SMC gold cohort에 포함하지 않은 독립 external test set", ha="center", va="center", fontsize=13, color="#4a5964")

    add_card(ax, 0.65, 4.42, 4.65, 2.95, he, "H&E WSI", "15 biopsy event: 비거부 5 / ACR 5 / AMR 5", "#cce9e6")
    add_card(ax, 5.68, 4.42, 4.65, 2.95, ihc, "IHC WSI", "45 WSI: CD3, CD8, CD68 (각 biopsy에 paired)", "#d9e6f7")
    add_card(ax, 10.7, 4.42, 4.65, 2.95, spatial, "Xenium spatial tissue", "2개 whole-tissue image에서 195개 core ROI", "#f8dfc8")

    box(ax, 0.95, 2.26, 8.95, 1.04, "WSI event-level evaluation\nH&E 단독 또는 H&E + IHC를 동일 biopsy event로 집계", "#e6f4f2", fontsize=14)
    box(ax, 10.95, 2.26, 4.10, 1.04, "Core-level evaluation\n62명, 195개 H&E core ROI", "#fcebdd", fontsize=14)
    arrow(ax, (2.95, 4.42), (3.85, 3.30), "#258f8b")
    arrow(ax, (7.95, 4.42), (7.0, 3.30), "#4781bc")
    arrow(ax, (13.0, 4.42), (13.0, 3.30), "#c87531")

    box(ax, 2.7, 0.72, 10.6, 0.76, "조직영역 mask → 물리 해상도 기반 patch → UNI-v2 feature → internal model inference", "#e9e5f5", fontsize=14)
    arrow(ax, (5.43, 2.26), (6.5, 1.48))
    arrow(ax, (13.0, 2.26), (10.2, 1.48))
    ax.text(8, 0.26, "보고 시 WSI와 core는 표본 단위와 독립성이 달라 성능을 분리하여 제시", ha="center", fontsize=12, color="#4a5964")

    fig.tight_layout(pad=0)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=220, facecolor="white")


if __name__ == "__main__":
    main()
