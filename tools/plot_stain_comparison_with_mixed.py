#!/usr/bin/env python3
"""Create a presentation table comparing mixed, H&E-only, and IHC-only runs."""

from pathlib import Path
import re

import matplotlib.pyplot as plt
from matplotlib import font_manager
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = ROOT / "results" / "smc_cv_comparison"
MIXED_CSV = RESULT_DIR / "averaged_summary (3).csv"
STAIN_CSV = RESULT_DIR / "averaged_summary (4).csv"
OUTPUT_CSV = RESULT_DIR / "stain_comparison_with_unfiltered_mixed.csv"
OUTPUT_PNG = RESULT_DIR / "stain_comparison_with_unfiltered_mixed.png"

TASKS = {
    "ACR": "acr_0r_vs_rest",
    "AMR": "amr_positive",
}
MAGNIFICATIONS = ["40x", "20x", "10x", "5x"]

# Previously calculated after subtracting each stain cohort's positive prevalence.
ADJUSTED_DELTA_AUPRC = {
    ("ACR", "40x"): 0.029,
    ("ACR", "20x"): 0.021,
    ("ACR", "10x"): 0.070,
    ("ACR", "5x"): 0.137,
    ("AMR", "40x"): 0.046,
    ("AMR", "20x"): 0.160,
    ("AMR", "10x"): 0.063,
    ("AMR", "5x"): 0.179,
}


def select_row(df: pd.DataFrame, task_fragment: str, magnification: str, cohort=None):
    pattern = rf"^smc_{re.escape(task_fragment)}_.*_{re.escape(magnification)}_"
    mask = df["experiment"].str.contains(pattern, regex=True)
    if cohort:
        mask &= df["experiment"].str.contains(rf"_{cohort}_s1$")
    else:
        mask &= ~df["experiment"].str.contains(
            r"_(?:he_only|ihc_only|mixed_known|non_he)_s1$", regex=True
        )
    rows = df.loc[mask]
    if len(rows) != 1:
        raise ValueError(
            f"Expected one row for {task_fragment}/{magnification}/{cohort}, found {len(rows)}"
        )
    return rows.iloc[0]


def build_table() -> pd.DataFrame:
    mixed = pd.read_csv(MIXED_CSV)
    stain = pd.read_csv(STAIN_CSV)
    records = []
    for task, fragment in TASKS.items():
        for mag in MAGNIFICATIONS:
            mixed_row = select_row(mixed, fragment, mag)
            he_row = select_row(stain, fragment, mag, "he_only")
            ihc_row = select_row(stain, fragment, mag, "ihc_only")
            records.append(
                {
                    "Task": task,
                    "배율": mag,
                    "Mixed AUROC": mixed_row["auroc_mean"],
                    "H&E AUROC": he_row["auroc_mean"],
                    "IHC AUROC": ihc_row["auroc_mean"],
                    "ΔAUROC (IHC-H&E)": ihc_row["auroc_mean"] - he_row["auroc_mean"],
                    "Mixed AUPRC": mixed_row["average_precision_mean"],
                    "H&E AUPRC": he_row["average_precision_mean"],
                    "IHC AUPRC": ihc_row["average_precision_mean"],
                    "보정 ΔAUPRC (IHC-H&E)": ADJUSTED_DELTA_AUPRC[(task, mag)],
                }
            )
    return pd.DataFrame(records)


def choose_font():
    candidates = ["Malgun Gothic", "Noto Sans CJK KR", "NanumGothic", "Arial"]
    installed = {font.name for font in font_manager.fontManager.ttflist}
    for candidate in candidates:
        if candidate in installed:
            plt.rcParams["font.family"] = candidate
            break
    plt.rcParams["axes.unicode_minus"] = False


def render_table(df: pd.DataFrame):
    choose_font()
    display = df.copy()
    metric_cols = [
        "Mixed AUROC", "H&E AUROC", "IHC AUROC",
        "Mixed AUPRC", "H&E AUPRC", "IHC AUPRC",
    ]
    delta_cols = ["ΔAUROC (IHC-H&E)", "보정 ΔAUPRC (IHC-H&E)"]
    for col in metric_cols:
        display[col] = display[col].map(lambda value: f"{value:.3f}")
    for col in delta_cols:
        display[col] = display[col].map(lambda value: f"{value:+.3f}")

    labels = [
        "Task", "배율", "Mixed\nAUROC", "H&E\nAUROC", "IHC\nAUROC",
        "ΔAUROC\n(IHC-H&E)", "Mixed\nAUPRC", "H&E\nAUPRC", "IHC\nAUPRC",
        "보정 ΔAUPRC\n(IHC-H&E)",
    ]
    fig, ax = plt.subplots(figsize=(17, 5.3), dpi=180)
    ax.axis("off")
    table = ax.table(
        cellText=display.values,
        colLabels=labels,
        cellLoc="center",
        colLoc="center",
        loc="center",
        colWidths=[0.075, 0.055, 0.098, 0.09, 0.09, 0.115, 0.098, 0.09, 0.09, 0.14],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 1.65)

    for col in range(len(labels)):
        cell = table[(0, col)]
        cell.set_facecolor("#33383f")
        cell.get_text().set_color("white")
        cell.get_text().set_weight("bold")
        cell.set_height(0.15)

    for row_idx, (_, row) in enumerate(df.iterrows(), start=1):
        fill = "#eef4f8" if row["Task"] == "ACR" else "#fff2e8"
        for col in range(len(labels)):
            table[(row_idx, col)].set_facecolor(fill)
            table[(row_idx, col)].set_edgecolor("#6c737a")

        auroc_values = row[["Mixed AUROC", "H&E AUROC", "IHC AUROC"]].to_numpy()
        auprc_values = row[["Mixed AUPRC", "H&E AUPRC", "IHC AUPRC"]].to_numpy()
        table[(row_idx, 2 + int(auroc_values.argmax()))].get_text().set_weight("bold")
        table[(row_idx, 6 + int(auprc_values.argmax()))].get_text().set_weight("bold")

    fig.text(
        0.5, 0.955, "염색 구성별 교차검증 성능 비교",
        ha="center", va="top", fontsize=17, fontweight="bold", color="#202428",
    )
    fig.text(
        0.5, 0.905,
        "Mixed = 염색 필터를 적용하지 않은 전체 weakunique3 데이터셋 · Δ 지표 = IHC - H&E",
        ha="center", va="top", fontsize=10.5, color="#4f5963",
    )
    fig.tight_layout(rect=(0.01, 0.02, 0.99, 0.88))
    fig.savefig(OUTPUT_PNG, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    df = build_table()
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    render_table(df)
    print(df.to_string(index=False, float_format=lambda value: f"{value:.3f}"))
    print(f"[OK] CSV: {OUTPUT_CSV}")
    print(f"[OK] PNG: {OUTPUT_PNG}")


if __name__ == "__main__":
    main()
