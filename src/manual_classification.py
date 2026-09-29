"""Interactive track classification (correct/incorrect) on top of a Napari viewer."""

from __future__ import annotations

from typing import TYPE_CHECKING

import napari  # type: ignore[import-untyped]
import napari.layers  # type: ignore[import-untyped]
import napari.utils.colormaps  # type: ignore[import-untyped]
import napari.utils.colormaps.standardize_color  # type: ignore[import-untyped]
import numpy as np
from byotrack.napari.utils import tracks_to_napari_tracks
from napari.components._viewer_mouse_bindings import double_click_to_zoom  # type: ignore[import-untyped]

if TYPE_CHECKING:
    from collections.abc import Collection

    import byotrack

_GREEN = napari.utils.colormaps.standardize_color.transform_color("green")[0]
_RED = napari.utils.colormaps.standardize_color.transform_color("red")[0]


class _TrackClassifier:
    """Holds the interactive state backing `add_track_classifier`.

    Builds the "Tracked points"/"Tracks" layers, keeps the point-level arrays needed to resolve a
    double-click into the closest track at the current frame, and updates the layers' colors (and
    `correct`) when a track's correctness is toggled.

    Attributes:
        correct (dict[int, bool]): Mapping of `track.identifier` to its correctness. This is
            the single source of truth: `point_correct`/`point_colors` are derived from it.

    """

    def __init__(
        self,
        viewer: napari.Viewer,
        tracks: Collection[byotrack.Track],
        correct: dict[int, bool] | None = None,
        *,
        anisotropy: tuple[float, float, float],
        track_width: float,
        max_distance: float,
    ) -> None:
        self.viewer = viewer
        self.max_distance = max_distance
        self.correct = correct or {}

        for track in tracks:
            self.correct[track.identifier] = self.correct.get(track.identifier, True)

        dim = next(iter(tracks)).dim
        self.anisotropy = np.asarray(anisotropy[-dim:], dtype=np.float32)
        axis_labels: tuple[str, ...] = ("Time", "Depth", "Height", "Width") if dim == 3 else ("Time", "Height", "Width")
        scale = (1.0, *anisotropy[-dim:])

        points, parents, features_points = tracks_to_napari_tracks(
            tracks, {"correct": {identifier: int(value) for identifier, value in self.correct.items()}}
        )
        self.points = points
        self.point_track_ids = points[:, 0].astype(np.int64)
        self.point_frames = points[:, 1].astype(np.int64)
        self.point_positions_scaled = points[:, 2:] * self.anisotropy
        self.point_colors = np.array([_RED, _GREEN])[features_points["correct"]]

        self.features_points: dict[str, np.ndarray] = {
            "time": self.point_frames,
            **features_points,
        }

        self.points_layer = napari.layers.Points(
            points[:, 1:],
            name="Tracked points",
            size=track_width,
            face_color=self.point_colors,
            axis_labels=axis_labels,
            scale=scale,
            blending="additive",
        )

        self.tracks_layer = napari.layers.Tracks(
            points,
            name="Tracks",
            graph=parents,
            features=self.features_points,
            colormaps_dict={"correct": napari.utils.colormaps.Colormap(["red", "green"], name="correct")},
            color_by="correct",
            tail_width=track_width,
            axis_labels=axis_labels,
            scale=scale,
        )

        viewer.add_layer(self.points_layer)
        viewer.add_layer(self.tracks_layer)

        viewer.mouse_double_click_callbacks.append(self._on_double_click)

    def _set_correct(self, track_id: int, *, is_correct: bool) -> None:
        """Update `track_correct` for `track_id` and refresh both layers' colors accordingly."""
        self.correct[track_id] = is_correct
        rows = self.point_track_ids == track_id

        self.point_colors[rows] = _GREEN if is_correct else _RED
        self.points_layer.face_color = self.point_colors

        self.features_points["correct"][rows] = 1 if is_correct else 0
        self.tracks_layer.features = self.features_points
        self.tracks_layer.color_by = self.tracks_layer.color_by  # Force a recolor (features alone doesn't)

    def _on_double_click(self, viewer: napari.Viewer, event) -> None:
        """Toggle the correctness of the closest track to the double-click, at the current frame."""
        data_pos = self.points_layer.world_to_data(event.position)

        frame = round(viewer.dims.current_step[0])
        mask = self.point_frames == frame
        if not np.any(mask):
            return

        click_scaled = np.asarray(data_pos[1:], dtype=np.float32) * self.anisotropy
        dists = np.linalg.norm(self.point_positions_scaled[mask] - click_scaled, axis=-1)
        best_local = int(np.argmin(dists))
        if dists[best_local] > self.max_distance:
            return

        track_id = int(self.point_track_ids[mask][best_local])
        self._set_correct(track_id, is_correct=not self.correct[track_id])  # Toggles


def add_track_classifier(
    viewer: napari.Viewer,
    tracks: Collection[byotrack.Track],
    correct: dict[int, bool] | None = None,
    *,
    anisotropy: tuple[float, float, float] = (1.0, 1.0, 1.0),
    track_width: float = 5.0,
    max_distance: float | None = None,
    disable_double_click_zoom: bool = True,
) -> dict[int, bool]:
    """Add tracks to the Napari viewer, with double-click callback for track classification.

    Adds a "Tracked points" points layer and a "Tracks" tracks layer (like `add_tracks`), colored
    green ("correct") or red ("incorrect") per track. Double-clicking selects the closest tracked
    point to the click, at the currently displayed frame, and toggles its track's correctness for
    all of its timepoints (both layers are recolored immediately). Double-clicks farther than
    `max_distance` (in world/scaled units) from every point on the current frame are ignored.

    Note:
        For 3D tracks displayed as a single 2D z-slice (`viewer.dims.ndisplay == 2`), a click is
        matched against every point at the current *time* frame, regardless of the currently
        displayed z-slice (only the time axis is used to filter candidates).

    Args:
        viewer (napari.Viewer): Napari viewer to add the tracks to.
        tracks (Collection[byotrack.Track]): Tracks to display and classify.
        correct (dict[int, bool] | None): Optional pre-classification of the tracks, mapping
            `track.identifier` to its correctness (True: correct).
            Mutated in place.
            Default: None (All tracks are correct)
        anisotropy (tuple[float, float, float]): Spatial anisotropy ([Z, ]Y, X) used to scale the
            layers, and to convert `max_distance` and click positions into consistent world units.
            Default: (1.0, 1.0, 1.0)
        track_width (float): Size of the tracked points and width of the track trails.
            Default: 5.0
        max_distance (float | None): Maximum distance (in world/scaled units) between a double-click
            and the closest tracked point, at the current frame, for the click to toggle that
            track's correctness. Farther clicks are a no-op.
            Default: None (defaults to `2 * track_width`)
        disable_double_click_zoom (bool): Napari zooms the camera on double-click by default, which
            would otherwise also fire on every classification click. If True, remove that default
            behaviour from `viewer.mouse_double_click_callbacks`.
            Default: True

    Returns:
        dict[int, bool]: Mapping of `track.identifier` to its correctness (True: correct).
            Can be mutated in place by the viewer, as long as it is not closed.

    """
    if max_distance is None:
        max_distance = 2 * track_width

    if disable_double_click_zoom and double_click_to_zoom in viewer.mouse_double_click_callbacks:
        viewer.mouse_double_click_callbacks.remove(double_click_to_zoom)

    classifier = _TrackClassifier(
        viewer, tracks, correct, anisotropy=anisotropy, track_width=track_width, max_distance=max_distance
    )

    return classifier.correct
