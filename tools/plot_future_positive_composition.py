#!/usr/bin/env python3
"""Plot future-positive biopsy pairs by baseline status."""

from pathlib import Path

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt


OUTPUT = Path("results/smc_cv_comparison/future_positive_composition.png")
LABELS = ["ACR ≥2R", "AMR 양성"]
INCIDENT = [8, 6]
BASELINE_POSITIVE = [1, 5]
TOTAL_PAIRS = 442


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
    totals = [incident + existing for incident, existing in zip(INCIDENT, BASELINE_POSITIVE)]

    fig, ax = plt.subplots(figsize=(8.8, 6.2), dpi=180)
    x = range(len(LABELS))
    width = 0.56
    incident_bars = ax.bar(
        x, INCIDENT, width=width, color="#26747A", edgecolor="#174B50",
        linewidth=0.8, label="기저 음성에서 새로 발생",
    )
    existing_bars = ax.bar(
        x, BASELINE_POSITIVE, width=width, bottom=INCIDENT, color="#D9895B",
        edgecolor="#955333", linewidth=0.8, label="기저 양성에서 미래에도 양성",
    )

    ax.set_title("미래 양성 생검 쌍의 구성", fontsize=19, fontweight="bold", pad=18)
    ax.set_ylabel("연속 생검 쌍 수", fontsize=12, labelpad=10)
    ax.set_xticks(list(x), LABELS)
    ax.set_xlim(-0.65, 1.65)
    ax.set_ylim(0, 13.5)
    ax.grid(axis="y", color="#D9DEE2", linewidth=0.7, alpha=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#737B80")
    ax.tick_params(axis="both", labelsize=11, length=0)

    for i, (incident, existing, total) in enumerate(zip(INCIDENT, BASELINE_POSITIVE, totals)):
        ax.text(i, incident / 2, f"{incident}\n({incident / TOTAL_PAIRS:.1%})",
                ha="center", va="center", color="white", fontsize=11, fontweight="bold")
        ax.text(i, incident + existing / 2, f"{existing}\n({existing / TOTAL_PAIRS:.1%})",
                ha="center", va="center", color="#352017", fontsize=10, fontweight="bold")
        ax.text(i, total + 0.45, f"전체 양성 {total}/{TOTAL_PAIRS} ({total / TOTAL_PAIRS:.1%})",
                ha="center", va="bottom", fontsize=11, fontweight="bold", color="#202528")

    ax.legend(loc="upper right", frameon=False, fontsize=10)
    fig.text(
        0.5, 0.025,
        "동일 환자의 연속 생검 쌍 442개를 공통 분모로 사용",
        ha="center", fontsize=10, color="#555D61",
    )
    fig.subplots_adjust(left=0.12, right=0.97, top=0.85, bottom=0.16)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(OUTPUT.resolve())


if __name__ == "__main__":
    main()
