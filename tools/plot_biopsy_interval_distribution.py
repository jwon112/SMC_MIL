#!/usr/bin/env python3
"""Plot the distribution of intervals between consecutive biopsies."""

from pathlib import Path

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt


OUTPUT = Path("results/smc_cv_comparison/biopsy_interval_distribution.png")
LABELS = ["1-30일", "31-90일", "91-180일", "181-365일", "365일 초과"]
COUNTS = [121, 215, 81, 21, 4]
PERCENTAGES = [27.4, 48.6, 18.3, 4.8, 0.9]


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
    fig, ax = plt.subplots(figsize=(10.5, 6.2), dpi=180)
    bars = ax.bar(LABELS, COUNTS, width=0.66, color="#26747A", edgecolor="#174B50", linewidth=0.8)

    ax.set_title("연속 생검 간격 분포", fontsize=19, fontweight="bold", pad=18)
    ax.set_xlabel("생검 간격", fontsize=12, labelpad=12)
    ax.set_ylabel("연속 생검 쌍 수", fontsize=12, labelpad=10)
    ax.set_ylim(0, 250)
    ax.grid(axis="y", color="#D9DEE2", linewidth=0.7, alpha=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#737B80")
    ax.tick_params(axis="both", labelsize=11, length=0)

    for bar, count, percentage in zip(bars, COUNTS, PERCENTAGES):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 6,
            f"{count} ({percentage:.1f}%)",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
            color="#202528",
        )

    fig.text(0.5, 0.025, "동일 환자에서 시간순으로 인접한 두 생검을 한 쌍으로 정의함 (총 442쌍)",
             ha="center", fontsize=10, color="#555D61")
    fig.subplots_adjust(left=0.11, right=0.97, top=0.86, bottom=0.20)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(OUTPUT.resolve())


if __name__ == "__main__":
    main()
