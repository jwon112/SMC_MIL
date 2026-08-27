"""ipywidgets quality-review UI for the WSI curation queue."""

from __future__ import annotations

from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path

import pandas as pd
from IPython.display import display
from PIL import Image
import ipywidgets as widgets


QUALITY_STATUSES = ("usable", "usable_low_quality", "exclude")


def _read_queue(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Quality review queue not found: {path}")
    frame = pd.read_csv(path, dtype=str).fillna("")
    required = {"slide_id", "thumbnail_path", "quality_manual_status", "quality_auto_flags"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Quality review queue missing columns: {sorted(missing)}")
    if frame.slide_id.duplicated().any():
        raise ValueError("Quality review queue contains duplicate slide_id values")
    return frame


def _thumbnail_bytes(path: Path) -> tuple[bytes, str]:
    with Image.open(path) as source:
        image = source.convert("RGB")
        image.thumbnail((1500, 1500))
        buffer = BytesIO()
        image.save(buffer, format="JPEG", quality=92)
    return buffer.getvalue(), "jpeg"


class QualityReviewEditor:
    def __init__(self, queue_path: Path, reviewer: str = "") -> None:
        self.queue_path = queue_path
        self.frame = _read_queue(queue_path)
        self.index = self._first_pending_index()

        self.title = widgets.HTML()
        self.metadata = widgets.HTML()
        self.image = widgets.Image(
            format="jpeg",
            layout=widgets.Layout(width="100%", height="760px", object_fit="contain", border="1px solid #bbb"),
        )
        self.image_status = widgets.HTML()
        self.reason = widgets.Textarea(
            placeholder="Optional rationale, e.g. severe grid artifact / technically interpretable but pale",
            layout=widgets.Layout(width="100%", height="76px"),
        )
        self.reviewer = widgets.Text(value=reviewer, placeholder="Reviewer", description="Reviewer")
        self.position = widgets.BoundedIntText(min=1, max=max(1, len(self.frame)), description="Slide")
        self.previous_button = widgets.Button(description="Previous")
        self.skip_button = widgets.Button(description="Skip")
        self.usable_button = widgets.Button(description="Usable", button_style="success")
        self.low_quality_button = widgets.Button(description="Usable low quality", button_style="warning")
        self.exclude_button = widgets.Button(description="Exclude", button_style="danger")
        self.clear_button = widgets.Button(description="Clear decision")
        self.message = widgets.HTML()

        self.previous_button.on_click(lambda _: self.move(-1))
        self.skip_button.on_click(lambda _: self.move(1))
        self.usable_button.on_click(lambda _: self.save_and_move("usable"))
        self.low_quality_button.on_click(lambda _: self.save_and_move("usable_low_quality"))
        self.exclude_button.on_click(lambda _: self.save_and_move("exclude"))
        self.clear_button.on_click(lambda _: self.clear_decision())
        self.position.observe(self.go_to_position, names="value")

    def _first_pending_index(self) -> int:
        pending = self.frame.index[self.frame["quality_manual_status"].eq("")]
        return int(pending[0]) if len(pending) else 0

    def _write(self) -> None:
        temporary = self.queue_path.with_suffix(".csv.tmp")
        self.frame.to_csv(temporary, index=False)
        temporary.replace(self.queue_path)

    def render(self) -> None:
        row = self.frame.iloc[self.index]
        completed = int(self.frame["quality_manual_status"].isin(QUALITY_STATUSES).sum())
        self.title.value = (
            f"<h3>{self.index + 1} / {len(self.frame)} - {row.slide_id}</h3>"
            f"<b>Completed:</b> {completed} / {len(self.frame)}"
        )
        flags = row.quality_auto_flags or "random_audit"
        self.metadata.value = (
            f"<b>Dataset:</b> {row.get('source_dataset', '')} &nbsp; "
            f"<b>Original mask QC:</b> {row.get('qc_status', '')} &nbsp; "
            f"<b>Review trigger:</b> {flags}<br>"
            f"<b>Mask source:</b> {row.get('tissue_pixel_source', '')} &nbsp; "
            f"<b>Tissue ratio:</b> {row.get('tissue_ratio', '')} &nbsp; "
            f"<b>L0 patches:</b> {row.get('patch_count_l0', '')} ({row.get('tissue_adequacy_auto', '')}) &nbsp; "
            f"<b>Sharpness:</b> {row.get('sharpness_score', '')} &nbsp; "
            f"<b>Grid score:</b> {row.get('grid_periodicity_score', '')}"
        )
        self.reason.value = row.get("quality_manual_reason", "")
        self.position.unobserve(self.go_to_position, names="value")
        self.position.value = self.index + 1
        self.position.observe(self.go_to_position, names="value")
        source = Path(row.thumbnail_path)
        if source.is_file():
            try:
                self.image.value, self.image.format = _thumbnail_bytes(source)
                self.image_status.value = ""
            except Exception as exc:
                self.image.value = b""
                self.image_status.value = f"<b>Could not load thumbnail:</b> {type(exc).__name__}"
        else:
            self.image.value = b""
            self.image_status.value = "<b>Thumbnail missing.</b> Review as exclude only if it cannot be regenerated."

    def go_to_position(self, change: dict) -> None:
        self.index = int(change["new"]) - 1
        self.render()

    def move(self, offset: int) -> None:
        self.index = max(0, min(len(self.frame) - 1, self.index + offset))
        self.render()

    def save_and_move(self, status: str) -> None:
        if status not in QUALITY_STATUSES:
            raise ValueError(status)
        self.frame.loc[self.frame.index[self.index], "quality_manual_status"] = status
        self.frame.loc[self.frame.index[self.index], "quality_manual_reason"] = self.reason.value.strip()
        self.frame.loc[self.frame.index[self.index], "quality_reviewer"] = self.reviewer.value.strip()
        self.frame.loc[self.frame.index[self.index], "quality_reviewed_at"] = datetime.now(timezone.utc).isoformat()
        self._write()
        self.message.value = f"<b>Saved:</b> {status}"
        self.move(1)

    def clear_decision(self) -> None:
        self.frame.loc[self.frame.index[self.index], [
            "quality_manual_status", "quality_manual_reason", "quality_reviewer", "quality_reviewed_at",
        ]] = ""
        self._write()
        self.message.value = "<b>Cleared current decision.</b>"
        self.render()

    def show(self) -> None:
        controls = widgets.HBox([
            self.previous_button, self.skip_button, self.usable_button,
            self.low_quality_button, self.exclude_button, self.clear_button,
        ], layout=widgets.Layout(flex_flow="row wrap", gap="8px"))
        display(widgets.VBox([
            self.title,
            self.metadata,
            widgets.HBox([self.position, self.reviewer]),
            self.image_status,
            self.image,
            self.reason,
            controls,
            self.message,
        ], layout=widgets.Layout(width="100%")))
        self.render()


def launch_quality_review(curation_root: str | Path, reviewer: str = "") -> QualityReviewEditor:
    """Display an in-notebook editor that writes each decision to the queue CSV."""
    root = Path(curation_root)
    editor = QualityReviewEditor(root / "quality_review_queue.csv", reviewer=reviewer)
    editor.show()
    return editor
