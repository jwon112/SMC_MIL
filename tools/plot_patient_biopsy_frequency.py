#!/usr/bin/env python3
"""Plot the patient distribution by number of biopsy events."""

from pathlib import Path

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt


OUTPUT = Path("results/smc_cv_comparison/patient_biopsy_frequency_distribution.png")
LABELS = ["1회", "2회", "3회", "4회", "5회 이상"]
COUNTS = [26, 13, 17, 16, 64]
TOTAL = sum(COUNTS)


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
    percentages = [count / TOTAL * 100 for count in COUNTS]

    fig, ax = plt.subplots(figsize=(10.5, 6.2), dpi=180)
    bars = ax.bar(
        LABELS,
        COUNTS,
        width=0.66,
        color="#26747A",
        edgecolor="#174B50",
        linewidth=0.8,
    )

    ax.set_title("환자별 생검 횟수 분포", fontsize=19, fontweight="bold", pad=18)
    ax.set_xlabel("생검 event 수", fontsize=12, labelpad=12)
    ax.set_ylabel("환자 수", fontsize=12, labelpad=10)
    ax.set_ylim(0, 75)
    ax.grid(axis="y", color="#D9DEE2", linewidth=0.7, alpha=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#737B80")
    ax.tick_params(axis="both", labelsize=11, length=0)

    for bar, count, percentage in zip(bars, COUNTS, percentages):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1.8,
            f"{count}명 ({percentage:.1f}%)",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
            color="#202528",
        )

    fig.text(
        0.5,
        0.025,
        "환자별 병리번호가 매칭된 생검 event 기준 (총 136명)",
        ha="center",
        fontsize=10,
        color="#555D61",
    )
    fig.subplots_adjust(left=0.11, right=0.97, top=0.86, bottom=0.20)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(OUTPUT.resolve())


if __name__ == "__main__":
    main()
