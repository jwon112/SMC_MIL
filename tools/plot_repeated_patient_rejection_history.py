#!/usr/bin/env python3
"""Plot rejection-history phenotypes among patients with repeated biopsies."""

from pathlib import Path

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


OUTPUT = Path("results/smc_cv_comparison/repeated_patient_rejection_history.png")
TOTAL = 110
CATEGORIES = [
    ("음성 이력", 90, "#66747A"),
    ("ACR ≥2R 이력만 있음", 7, "#26747A"),
    ("AMR ≥pAMR1 이력만 있음", 13, "#D9895B"),
    ("두 조건 모두 있음", 0, "#8A6FA8"),
]


def configure_font() -> None:
    candidates = [
        Path("C:/Windows/Fonts/malgun.ttf"),
        Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
    ]
    for path in candidates:
        if path.is_file():
            fm.fontManager.addfont(path)
            plt.rcParams["font.family"] = fm.FontProperties(fname=path).get_name()
            break
    plt.rcParams["axes.unicode_minus"] = False


def main() -> None:
    configure_font()
    fig, ax = plt.subplots(figsize=(11.5, 4.8), dpi=180)

    left = 0.0
    for label, count, color in CATEGORIES:
        percentage = count / TOTAL * 100
        if count:
            ax.barh(0, percentage, left=left, height=0.42, color=color,
                    edgecolor="white", linewidth=1.5)
            ax.text(left + percentage / 2, 0, f"{count}명\n({percentage:.1f}%)",
                    ha="center", va="center", color="white", fontsize=11,
                    fontweight="bold")
        left += percentage

    positive_start = CATEGORIES[0][1] / TOTAL * 100
    bracket_y = 0.34
    ax.plot([positive_start, 100], [bracket_y, bracket_y], color="#33383B", lw=1.3)
    ax.vlines([positive_start, 100], bracket_y - 0.05, bracket_y + 0.05,
              color="#33383B", lw=1.3)
    ax.text((positive_start + 100) / 2, 0.58, "전체 양성 20명 (18.2%)",
            ha="center", va="bottom", fontsize=12, fontweight="bold")

    ax.set_title("반복 생검 환자의 거부반응 이력 분포", fontsize=19,
                 fontweight="bold", pad=18)
    ax.set_xlim(0, 100)
    ax.set_ylim(-0.48, 0.95)
    ax.set_yticks([])
    ax.set_xlabel("환자 비율 (%)", fontsize=12, labelpad=10)
    ax.set_xticks(range(0, 101, 20))
    ax.grid(axis="x", color="#D9DEE2", linewidth=0.7, alpha=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color("#737B80")
    ax.tick_params(axis="x", labelsize=10, length=0)

    legend = [
        Patch(facecolor=color, label=f"{label}: {count}명 ({count / TOTAL:.1%})")
        for label, count, color in CATEGORIES
    ]
    ax.legend(handles=legend, loc="lower center", bbox_to_anchor=(0.5, -0.65),
              ncol=2, frameon=False, fontsize=10)

    fig.text(0.5, 0.02, "생검을 2회 이상 시행한 환자 기준 (n=110)",
             ha="center", fontsize=10, color="#555D61")
    fig.subplots_adjust(left=0.07, right=0.98, top=0.79, bottom=0.33)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(OUTPUT.resolve())


if __name__ == "__main__":
    main()
