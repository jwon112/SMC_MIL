#!/usr/bin/env python3
"""Create presentation figures for the conservative weak-linkage strategy."""

from pathlib import Path

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle


OUTPUT_DIR = Path("results/smc_cv_comparison/weak_linkage_figures")
TEAL = "#26747A"
ORANGE = "#D9895B"
DARK = "#202528"
GRAY = "#66747A"
LIGHT = "#EDF1F2"


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


def save_event_overlap_bar() -> None:
    total = 3396
    unique = 631
    overlap = 2765
    unique_pct = unique / total * 100
    overlap_pct = overlap / total * 100

    fig, ax = plt.subplots(figsize=(11.5, 4.7), dpi=180)
    ax.barh(0, unique_pct, height=0.42, color=TEAL, edgecolor="white", linewidth=1.5)
    ax.barh(0, overlap_pct, left=unique_pct, height=0.42, color=ORANGE,
            edgecolor="white", linewidth=1.5)
    ax.text(unique_pct / 2, 0, f"단일 환자 날짜\n{unique:,}건 ({unique_pct:.1f}%)",
            ha="center", va="center", color="white", fontsize=11, fontweight="bold")
    ax.text(unique_pct + overlap_pct / 2, 0,
            f"복수 환자 날짜\n{overlap:,}건 ({overlap_pct:.1f}%)",
            ha="center", va="center", color="white", fontsize=13, fontweight="bold")

    ax.set_title("EHR 생검 event의 생검일 중복 노출", fontsize=19, fontweight="bold", pad=18)
    ax.set_xlabel("전체 EHR 생검 event 중 비율 (%)", fontsize=12, labelpad=10)
    ax.set_xlim(0, 100)
    ax.set_ylim(-0.48, 0.48)
    ax.set_yticks([])
    ax.set_xticks(range(0, 101, 20))
    ax.grid(axis="x", color="#D9DEE2", linewidth=0.7, alpha=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color("#737B80")
    ax.tick_params(axis="x", labelsize=10, length=0)
    fig.text(0.5, 0.025,
             "복수 환자 날짜: 동일 날짜에 서로 다른 환자 2명 이상이 생검을 받은 경우 (n=3,396 events)",
             ha="center", fontsize=10, color="#555D61")
    fig.subplots_adjust(left=0.07, right=0.98, top=0.78, bottom=0.25)
    fig.savefig(OUTPUT_DIR / "ehr_event_date_overlap.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def save_multi_patient_date_distribution() -> None:
    labels = ["2명", "3명", "4명", "5명", "6명 이상"]
    counts = [266, 197, 143, 102, 86]
    total = sum(counts)
    percentages = [count / total * 100 for count in counts]

    fig, ax = plt.subplots(figsize=(10.5, 6.2), dpi=180)
    bars = ax.bar(labels, counts, width=0.66, color=TEAL, edgecolor="#174B50", linewidth=0.8)
    ax.set_title("복수 환자 생검일의 환자 수 분포", fontsize=19, fontweight="bold", pad=18)
    ax.set_xlabel("동일 날짜에 생검한 환자 수", fontsize=12, labelpad=12)
    ax.set_ylabel("생검일 수", fontsize=12, labelpad=10)
    ax.set_ylim(0, 310)
    ax.grid(axis="y", color="#D9DEE2", linewidth=0.7, alpha=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#737B80")
    ax.tick_params(axis="both", labelsize=11, length=0)

    for bar, count, percentage in zip(bars, counts, percentages):
        ax.text(bar.get_x() + bar.get_width() / 2, count + 8,
                f"{count}일 ({percentage:.1f}%)", ha="center", va="bottom",
                fontsize=11, fontweight="bold", color=DARK)

    fig.text(0.5, 0.025,
             "동일 날짜에 2명 이상의 환자가 생검을 받은 794일 기준; 하루 최대 10명",
             ha="center", fontsize=10, color="#555D61")
    fig.subplots_adjust(left=0.11, right=0.97, top=0.86, bottom=0.20)
    fig.savefig(OUTPUT_DIR / "multi_patient_biopsy_date_distribution.png",
                bbox_inches="tight", facecolor="white")
    plt.close(fig)


def add_box(ax, center, width, height, text, facecolor, edgecolor="#405055",
            textcolor=DARK, fontsize=11, linewidth=1.3):
    x, y = center
    patch = Rectangle((x - width / 2, y - height / 2), width, height,
                      facecolor=facecolor, edgecolor=edgecolor, linewidth=linewidth)
    ax.add_patch(patch)
    ax.text(x, y, text, ha="center", va="center", color=textcolor,
            fontsize=fontsize, fontweight="bold", linespacing=1.35)


def add_diamond(ax, center, width, height, text):
    x, y = center
    points = [(x, y + height / 2), (x + width / 2, y),
              (x, y - height / 2), (x - width / 2, y)]
    ax.add_patch(Polygon(points, closed=True, facecolor="#F7F2E9",
                         edgecolor="#7B6A4D", linewidth=1.3))
    ax.text(x, y, text, ha="center", va="center", fontsize=10.5,
            fontweight="bold", color=DARK, linespacing=1.3)


def arrow(ax, start, end, label=None, label_offset=(0, 0)):
    ax.annotate("", xy=end, xytext=start,
                arrowprops={"arrowstyle": "-|>", "lw": 1.5, "color": "#455156"})
    if label:
        x = (start[0] + end[0]) / 2 + label_offset[0]
        y = (start[1] + end[1]) / 2 + label_offset[1]
        ax.text(x, y, label, ha="center", va="center", fontsize=10,
                fontweight="bold", color="#455156", backgroundcolor="white")


def save_decision_flow() -> None:
    fig, ax = plt.subplots(figsize=(12, 7), dpi=180)
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 10)
    ax.axis("off")
    ax.set_title("비매칭 WSI의 보수적 의사매칭 기준", fontsize=19,
                 fontweight="bold", pad=14)

    add_box(ax, (6, 8.9), 3.3, 0.9, "병리번호 비매칭 WSI", LIGHT)
    add_box(ax, (6, 7.35), 4.5, 1.05,
            "WSI scan date 이전 0-3일\nEHR 생검 후보 검색", "#E5F0F1")
    add_diamond(ax, (6, 5.55), 3.9, 1.5, "후보가 정확히\n1개인가?")
    add_box(ax, (2.25, 5.55), 2.8, 1.0, "매칭 제외\n(후보 없음/복수)", "#ECEDEF", textcolor="#515A5E")
    add_diamond(ax, (6, 3.4), 4.2, 1.6, "기존 gold의\n환자-생검일과 겹치는가?")
    add_box(ax, (2.25, 3.4), 3.1, 1.05, "possible_gold_extension\n보조 분석군", "#F8E9DF")
    add_box(ax, (6, 1.25), 3.7, 1.15, "weak_unique_new\n고신뢰 의사매칭군", TEAL,
            edgecolor="#174B50", textcolor="white", fontsize=11.5)

    arrow(ax, (6, 8.43), (6, 7.9))
    arrow(ax, (6, 6.82), (6, 6.3))
    arrow(ax, (4.05, 5.55), (3.7, 5.55), "아니오", (0, 0.28))
    arrow(ax, (6, 4.8), (6, 4.2), "예", (0.28, 0))
    arrow(ax, (3.9, 3.4), (3.8, 3.4), "예", (0, 0.28))
    arrow(ax, (6, 2.6), (6, 1.84), "아니오", (0.4, 0))

    ax.text(9.15, 5.55,
            "날짜 일치만으로는\n환자 특정 불가",
            ha="center", va="center", fontsize=11, color="#9A4F32",
            fontweight="bold")
    ax.plot([8.2, 10.1], [4.95, 4.95], color=ORANGE, lw=2)
    ax.text(6, 0.25,
            "확정 매칭이 아닌 제한된 메타데이터 기반 weak linkage",
            ha="center", fontsize=10, color="#555D61")

    fig.subplots_adjust(left=0.03, right=0.97, top=0.90, bottom=0.04)
    fig.savefig(OUTPUT_DIR / "weak_linkage_decision_flow.png",
                bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    configure_font()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    save_event_overlap_bar()
    save_multi_patient_date_distribution()
    save_decision_flow()
    for name in (
        "ehr_event_date_overlap.png",
        "multi_patient_biopsy_date_distribution.png",
        "weak_linkage_decision_flow.png",
    ):
        print((OUTPUT_DIR / name).resolve())


if __name__ == "__main__":
    main()
