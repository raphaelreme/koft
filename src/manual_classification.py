"""Interactive track classification (unassigned/correct/incorrect) on top of a Napari viewer."""

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

_GRAY = napari.utils.colormaps.standardize_color.transform_color("gray")[0]
_GREEN = napari.utils.colormaps.standardize_color.transform_color("green")[0]
_RED = napari.utils.colormaps.standardize_color.transform_color("red")[0]
_BLUE = napari.utils.colormaps.standardize_color.transform_color("blue")[0]

# Three classification states, cycled circularly on double-click:
#   unassigned (None) -> correct (True) -> incorrect (False) -> unassigned ...
# `_STATE_INDEX` maps each state to its index (0/1/2) into the color arrays below.
_STATE_INDEX: dict[bool | None, int] = {None: 0, True: 1, False: 2}
_NEXT_STATE: dict[bool | None, bool | None] = {None: True, True: False, False: None}

# Indexed by state index. Points encode the state on two channels: the face color carries the
# classification (gray/green/red), while a blue border flags the still-unassigned tracks.
_FACE_COLORS = np.stack([_GRAY, _GREEN, _RED])
_BORDER_COLORS = np.stack([_BLUE, _GRAY, _GRAY])


class _TrackClassifier:
    """Holds the interactive state backing `add_track_classifier`.

    Builds the "Tracked points"/"Tracks" layers, keeps the point-level arrays needed to resolve a
    double-click into the closest track at the current frame, and updates the layers' colors (and
    `correct`) when a track's state is cycled.

    Attributes:
        correct (dict[int, bool | None]): Mapping of `track.identifier` to its state, where
            `None` is unassigned, `True` is correct and `False` is incorrect. This is the single
            source of truth: the point face/border colors and the track feature are derived from it.

    """

    def __init__(
        self,
        viewer: napari.Viewer,
        tracks: Collection[byotrack.Track],
        correct: dict[int, bool | None] | None = None,
        *,
        anisotropy: tuple[float, float, float],
        track_width: float,
        max_distance: float,
    ) -> None:
        self.viewer = viewer
        self.max_distance = max_distance
        self.correct = correct or {}

        for track in tracks:
            self.correct[track.identifier] = self.correct.get(track.identifier, None)

        dim = next(iter(tracks)).dim
        self.anisotropy = np.asarray(anisotropy[-dim:], dtype=np.float32)
        axis_labels: tuple[str, ...] = ("Time", "Depth", "Height", "Width") if dim == 3 else ("Time", "Height", "Width")
        scale = (1.0, *anisotropy[-dim:])

        points, parents, features_points = tracks_to_napari_tracks(
            tracks, {"correct": {identifier: _STATE_INDEX[state] for identifier, state in self.correct.items()}}
        )
        self.points = points
        self.point_track_ids = points[:, 0].astype(np.int64)
        self.point_frames = points[:, 1].astype(np.int64)
        self.point_positions_scaled = points[:, 2:] * self.anisotropy

        # Per-point state index (0/1/2) drives the point face/border colors directly, while the
        # "correct" feature is rescaled to the colormap domain {0.0, 0.5, 1.0} so that the three
        # colors land exactly on the Tracks colormap's default controls [0, 0.5, 1].
        point_state = features_points["correct"].astype(np.int64)
        self.point_face_colors = _FACE_COLORS[point_state]
        self.point_border_colors = _BORDER_COLORS[point_state]
        features_points["correct"] = point_state.astype(np.float32) / 2.0

        self.features_points: dict[str, np.ndarray] = {
            "time": self.point_frames,
            **features_points,
        }

        self.points_layer = napari.layers.Points(
            points[:, 1:],
            name="Tracked points",
            size=track_width,
            face_color=self.point_face_colors,
            border_color=self.point_border_colors,
            axis_labels=axis_labels,
            scale=scale,
            blending="additive",
        )

        self.tracks_layer = napari.layers.Tracks(
            points,
            name="Tracks",
            graph=parents,
            features=self.features_points,
            colormaps_dict={"correct": napari.utils.colormaps.Colormap([_GRAY, _GREEN, _RED], name="correct")},
            color_by="correct",
            tail_width=track_width,
            axis_labels=axis_labels,
            scale=scale,
        )

        viewer.add_layer(self.points_layer)
        viewer.add_layer(self.tracks_layer)

        viewer.mouse_double_click_callbacks.append(self._on_double_click)

    def _set_state(self, track_id: int, state: bool | None) -> None:
        """Set `correct[track_id]` to `state` and refresh both layers' colors accordingly."""
        self.correct[track_id] = state
        idx = _STATE_INDEX[state]
        rows = self.point_track_ids == track_id

        self.point_face_colors[rows] = _FACE_COLORS[idx]
        self.point_border_colors[rows] = _BORDER_COLORS[idx]
        self.points_layer.face_color = self.point_face_colors
        self.points_layer.border_color = self.point_border_colors

        self.features_points["correct"][rows] = idx / 2.0
        self.tracks_layer.features = self.features_points
        self.tracks_layer.color_by = self.tracks_layer.color_by  # Force a recolor (features alone doesn't)

    def _on_double_click(self, viewer: napari.Viewer, event) -> None:
        """Cycle the state of the closest track to the double-click, at the current frame."""
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
        self._set_state(track_id, _NEXT_STATE[self.correct[track_id]])  # unassigned -> correct -> incorrect -> ...


def add_track_classifier(
    viewer: napari.Viewer,
    tracks: Collection[byotrack.Track],
    correct: dict[int, bool | None] | None = None,
    *,
    anisotropy: tuple[float, float, float] = (1.0, 1.0, 1.0),
    track_width: float = 5.0,
    max_distance: float | None = None,
    disable_double_click_zoom: bool = True,
) -> dict[int, bool | None]:
    """Add tracks to the Napari viewer, with double-click callback for track classification.

    Adds a "Tracked points" points layer and a "Tracks" tracks layer (like `add_tracks`). Each track
    has one of three states: unassigned (`None`, the default), correct (`True`) or incorrect
    (`False`). The point face color encodes the state (gray/green/red) and a blue point border flags
    the still-unassigned tracks; the track trail is colored gray/green/red to match. Double-clicking
    selects the closest tracked point to the click, at the currently displayed frame, and cycles its
    track's state for all of its timepoints (unassigned -> correct -> incorrect -> unassigned); both
    layers are recolored immediately. Double-clicks farther than `max_distance` (in world/scaled
    units) from every point on the current frame are ignored.

    Note:
        For 3D tracks displayed as a single 2D z-slice (`viewer.dims.ndisplay == 2`), a click is
        matched against every point at the current *time* frame, regardless of the currently
        displayed z-slice (only the time axis is used to filter candidates).

    Args:
        viewer (napari.Viewer): Napari viewer to add the tracks to.
        tracks (Collection[byotrack.Track]): Tracks to display and classify.
        correct (dict[int, bool | None] | None): Optional pre-classification of the tracks, mapping
            `track.identifier` to its state (None: unassigned, True: correct, False: incorrect).
            Mutated in place.
            Default: None (all tracks start unassigned)
        anisotropy (tuple[float, float, float]): Spatial anisotropy ([Z, ]Y, X) used to scale the
            layers, and to convert `max_distance` and click positions into consistent world units.
            Default: (1.0, 1.0, 1.0)
        track_width (float): Size of the tracked points and width of the track trails.
            Default: 5.0
        max_distance (float | None): Maximum distance (in world/scaled units) between a double-click
            and the closest tracked point, at the current frame, for the click to cycle that
            track's state. Farther clicks are a no-op.
            Default: None (defaults to `2 * track_width`)
        disable_double_click_zoom (bool): Napari zooms the camera on double-click by default, which
            would otherwise also fire on every classification click. If True, remove that default
            behaviour from `viewer.mouse_double_click_callbacks`.
            Default: True

    Returns:
        dict[int, bool | None]: Mapping of `track.identifier` to its state (None: unassigned,
            True: correct, False: incorrect).
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
