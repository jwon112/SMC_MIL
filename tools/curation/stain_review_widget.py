"""ipywidgets stain-group review UI for slides not resolved by filename rules."""

from __future__ import annotations

from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path

import pandas as pd
from IPython.display import display
from PIL import Image
import ipywidgets as widgets


STAIN_GROUPS = ("HE", "IHC", "special_other", "unknown")
DECISION_COLUMNS = ["slide_id", "stain_group", "stain_raw", "stain_confidence", "stain_note", "reviewed_at"]


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    return pd.read_csv(path, dtype=str).fillna("")


def _image_bytes(path: Path) -> tuple[bytes, str]:
    with Image.open(path) as source:
        image = source.convert("RGB")
        image.thumbnail((1400, 1400))
        buffer = BytesIO()
        image.save(buffer, format="JPEG", quality=92)
    return buffer.getvalue(), "jpeg"


class StainReviewEditor:
    def __init__(self, curation_root: Path, reviewer: str = "") -> None:
        self.curation_root = curation_root
        manifest = _read_csv(curation_root / "slide_curation_manifest.csv")
        required = {"slide_id", "thumbnail_path", "stain_group", "stain_signature"}
        missing = required.difference(manifest.columns)
        if missing:
            raise ValueError(f"Manifest missing columns: {sorted(missing)}")
        if manifest.slide_id.duplicated().any():
            raise ValueError("Manifest contains duplicate slide_id values")
        self.decisions_path = curation_root / "stain_slide_review.csv"
        decisions = self._load_decisions()
        items = manifest[manifest.stain_group.eq("unknown")].copy()
        self.items = items.merge(decisions, on="slide_id", how="left", validate="one_to_one", suffixes=("", "_decision"))
        for column in DECISION_COLUMNS[1:]:
            decision_column = f"{column}_decision"
            if decision_column in self.items:
                self.items[column] = self.items[decision_column].fillna("")
                self.items.drop(columns=decision_column, inplace=True)
            elif column not in self.items:
                self.items[column] = ""
        self.items.sort_values(["stain_color_cluster", "slide_id"], inplace=True, na_position="last")
        self.items.reset_index(drop=True, inplace=True)
        self.index = self._first_pending_index()

        self.title = widgets.HTML()
        self.metadata = widgets.HTML()
        self.image = widgets.Image(format="jpeg", layout=widgets.Layout(width="100%", height="760px", object_fit="contain"))
        self.cluster_image = widgets.Image(format="jpeg", layout=widgets.Layout(width="100%", height="760px", object_fit="contain"))
        self.image_status = widgets.HTML()
        self.stain_raw = widgets.Text(placeholder="Optional detail, e.g. Masson trichrome")
        self.confidence = widgets.Dropdown(options=[("High", "high"), ("Medium", "medium"), ("Low", "low")], value="medium")
        self.note = widgets.Textarea(placeholder="Optional note", layout=widgets.Layout(width="100%", height="72px"))
        self.reviewer = widgets.Text(value=reviewer, placeholder="Reviewer", description="Reviewer")
        self.position = widgets.BoundedIntText(min=1, max=max(1, len(self.items)), description="Slide")
        self.previous_button = widgets.Button(description="Previous")
        self.skip_button = widgets.Button(description="Skip")
        self.he_button = widgets.Button(description="H&E", button_style="success")
        self.ihc_button = widgets.Button(description="IHC", button_style="primary")
        self.special_button = widgets.Button(description="Special stain", button_style="warning")
        self.unknown_button = widgets.Button(description="Keep unknown")
        self.clear_button = widgets.Button(description="Clear decision")
        self.message = widgets.HTML()

        self.previous_button.on_click(lambda _: self.move(-1))
        self.skip_button.on_click(lambda _: self.move(1))
        self.he_button.on_click(lambda _: self.save_and_move("HE"))
        self.ihc_button.on_click(lambda _: self.save_and_move("IHC"))
        self.special_button.on_click(lambda _: self.save_and_move("special_other"))
        self.unknown_button.on_click(lambda _: self.save_and_move("unknown"))
        self.clear_button.on_click(lambda _: self.clear_decision())
        self.position.observe(self.go_to_position, names="value")

    def _load_decisions(self) -> pd.DataFrame:
        if not self.decisions_path.is_file():
            return pd.DataFrame(columns=DECISION_COLUMNS)
        decisions = _read_csv(self.decisions_path)
        missing = set(DECISION_COLUMNS[:2]).difference(decisions.columns)
        if missing:
            raise ValueError(f"Stain decisions missing columns: {sorted(missing)}")
        if decisions.slide_id.duplicated().any():
            raise ValueError("Stain decisions contain duplicate slide_id values")
        invalid = set(decisions.stain_group).difference(set(STAIN_GROUPS) | {""})
        if invalid:
            raise ValueError(f"Stain decisions contain unsupported groups: {sorted(invalid)}")
        for column in DECISION_COLUMNS:
            if column not in decisions:
                decisions[column] = ""
        return decisions[DECISION_COLUMNS]

    def _first_pending_index(self) -> int:
        pending = self.items.index[self.items.stain_group.eq("")]
        return int(pending[0]) if len(pending) else 0

    def _write(self) -> None:
        decisions = self.items[DECISION_COLUMNS].copy()
        decisions = decisions[decisions.stain_group.ne("")]
        temporary = self.decisions_path.with_suffix(".csv.tmp")
        decisions.to_csv(temporary, index=False)
        temporary.replace(self.decisions_path)

    @staticmethod
    def _cluster_path(root: Path, value: str) -> Path | None:
        try:
            return root / "stain_cluster_images" / f"cluster_{int(float(value)):02d}.jpg"
        except (TypeError, ValueError):
            return None

    def render(self) -> None:
        row = self.items.iloc[self.index]
        completed = int(self.items.stain_group.ne("").sum())
        self.title.value = (
            f"<h3>{self.index + 1} / {len(self.items)} - {row.slide_id}</h3>"
            f"<b>Reviewed:</b> {completed} / {len(self.items)}"
        )
        self.metadata.value = (
            f"<b>Signature:</b> {row.stain_signature} &nbsp; "
            f"<b>Color cluster:</b> {row.get('stain_color_cluster', '')} &nbsp; "
            f"<b>Dataset:</b> {row.get('source_dataset', '')}<br>"
            f"<b>Path:</b> {row.get('slide_rel_path', '')}"
        )
        self.stain_raw.value = row.stain_raw
        self.confidence.value = row.stain_confidence if row.stain_confidence in {"high", "medium", "low"} else "medium"
        self.note.value = row.stain_note
        self.position.unobserve(self.go_to_position, names="value")
        self.position.value = self.index + 1
        self.position.observe(self.go_to_position, names="value")

        source = Path(row.thumbnail_path)
        try:
            self.image.value, self.image.format = _image_bytes(source)
            self.image_status.value = ""
        except Exception as exc:
            self.image.value = b""
            self.image_status.value = f"<b>Could not load thumbnail:</b> {type(exc).__name__}"
        cluster_path = self._cluster_path(self.curation_root, row.get("stain_color_cluster", ""))
        try:
            if cluster_path and cluster_path.is_file():
                self.cluster_image.value, self.cluster_image.format = _image_bytes(cluster_path)
            else:
                self.cluster_image.value = b""
        except Exception:
            self.cluster_image.value = b""

    def go_to_position(self, change: dict) -> None:
        self.index = int(change["new"]) - 1
        self.render()

    def move(self, offset: int) -> None:
        self.index = max(0, min(len(self.items) - 1, self.index + offset))
        self.render()

    def save_and_move(self, stain_group: str) -> None:
        self.items.loc[self.index, "stain_group"] = stain_group
        self.items.loc[self.index, "stain_raw"] = self.stain_raw.value.strip()
        self.items.loc[self.index, "stain_confidence"] = self.confidence.value
        reviewer_note = self.note.value.strip()
        reviewer_prefix = f"reviewer={self.reviewer.value.strip()}" if self.reviewer.value.strip() else ""
        self.items.loc[self.index, "stain_note"] = "; ".join(value for value in [reviewer_prefix, reviewer_note] if value)
        self.items.loc[self.index, "reviewed_at"] = datetime.now(timezone.utc).isoformat()
        self._write()
        self.message.value = f"<b>Saved:</b> {stain_group}"
        self.move(1)

    def clear_decision(self) -> None:
        self.items.loc[self.index, DECISION_COLUMNS[1:]] = ""
        self._write()
        self.message.value = "<b>Cleared current decision.</b>"
        self.render()

    def show(self) -> None:
        controls = widgets.HBox([
            self.previous_button, self.skip_button, self.he_button, self.ihc_button,
            self.special_button, self.unknown_button, self.clear_button,
        ], layout=widgets.Layout(flex_flow="row wrap", gap="8px"))
        detail = widgets.HBox([
            widgets.VBox([widgets.HTML("<b>Current slide</b>"), self.image], layout=widgets.Layout(width="65%")),
            widgets.VBox([widgets.HTML("<b>Color-cluster examples</b>"), self.cluster_image], layout=widgets.Layout(width="35%")),
        ], layout=widgets.Layout(width="100%"))
        display(widgets.VBox([
            self.title,
            self.metadata,
            widgets.HBox([self.position, self.reviewer, self.confidence]),
            self.image_status,
            detail,
            self.stain_raw,
            self.note,
            controls,
            self.message,
        ], layout=widgets.Layout(width="100%")))
        self.render()


def launch_stain_review(curation_root: str | Path, reviewer: str = "") -> StainReviewEditor:
    """Display an editor for the unknown-stain subset and persist decisions on each click."""
    editor = StainReviewEditor(Path(curation_root), reviewer=reviewer)
    editor.show()
    return editor
