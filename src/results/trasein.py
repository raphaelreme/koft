from __future__ import annotations

import pathlib

import byotrack
import byotrack.napari
import napari  # type: ignore[import-untyped]
import napari.layers  # type: ignore[import-untyped]
import yaml
from magicgui import magicgui
from napari._vispy.filters.tracks import TracksFilter  # type: ignore[import-untyped]
from PIL import Image

from src.manual_classification import add_track_classifier

# Remove track shader

TracksFilter.VERT_SHADER = TracksFilter.VERT_SHADER.replace(
    "alpha = clamp(1.0-fade, 0.0, 1.0);",
    "alpha = (1.0 - fade) > 0.0 ? 1.0 : 0.0;",
)


def save_screenshot(
    viewer: napari.Viewer,
    output_path: pathlib.Path,
    frame: int = 50,
    track_width: int = 2,
    tail_length: int = 0,
    head_length: int = 300,
) -> None:
    viewer.dims.point = (frame, *viewer.dims.point[1:])  # Set temporal axis at the initial frame (0)
    viewer.layers[-2].visible = False  # Remove Points layer (only Image + Tracks layer now)
    layer: napari.layers.Tracks = viewer.layers[-1]
    layer.tail_width = track_width
    layer.tail_length = tail_length
    layer.head_length = head_length

    screenshot = viewer.export_figure(scale_factor=2)[..., :3]
    Image.fromarray(screenshot).save(output_path)


def main():
    experiment_path = pathlib.Path("experiment_folder") / "tracking" / "trasein" / "exp.0"
    video = byotrack.Video(yaml.safe_load((experiment_path / "config.yml").read_text())["data"]["trasein"]["video"])[
        ..., :1
    ].normalize()
    # detections_sequence = byotrack.Detections.load_multi_frames_detections(experiment_path / "detections")
    # det_per_frame = sum(len(detections) for detections in detections_sequence) / len(detections_sequence)
    # print(f"Reloaded {det_per_frame} detections per frame (avg)")

    if (experiment_path / "metrics.yml").exists():
        metrics: dict[str, dict[str, int | float]] = yaml.safe_load((experiment_path / "metrics.yml").read_text())
    else:
        metrics = {}

    print("-" * 100)
    print("Reloaded metrics:")
    print(yaml.safe_dump(metrics))
    print()
    print()

    for valid_path in experiment_path.glob("valid_*.yml"):
        method = valid_path.stem[len("valid_") :]
        tracks = byotrack.Track.load(experiment_path / f"tracks_{method}.pt")
        long_tracks = byotrack.Track.load(experiment_path / f"long_tracks_{method}.pt")
        classification = yaml.safe_load((experiment_path / f"valid_{method}.yml").read_text())

        print("-" * 50, method, "-" * 50)
        print(f"Reloaded {len(tracks)} tracks with {len(long_tracks)} long trajectories (> 90% tracked)")

        # Could add detections & tracks to debug
        # viewer = byotrack.napari.visualize(video, detections_sequence, tracks, run=False)
        # viewer.layers[-1].visible = False
        # viewer.layers[-2].visible = False
        # viewer.layers[-3].visible = False
        viewer = byotrack.napari.visualize(video, run=False, lazy=True)
        classification = add_track_classifier(viewer, long_tracks, correct=classification, max_distance=5.0)
        viewer._layer_slicer._force_sync = False  # noqa: SLF001  # Prevent lag
        # viewer.layers[-1].visible = True  # Disable if too much lag

        @magicgui(call_button="Save screenshot")
        def _screenshot(
            viewer=viewer,
            method=method,
            frame: int = 50,
            track_width: int = 2,
            tail_length: int = 0,
            head_length: int = 300,
        ) -> None:
            save_screenshot(viewer, experiment_path / f"{method}.png", frame, track_width, tail_length, head_length)
            print("Screenshot saved")

        viewer.window.add_dock_widget(_screenshot, area="right")

        napari.run()  # Modify in place `classification` from user inputs

        valid = len([k for k, v in classification.items() if v])
        invalid = len([k for k, v in classification.items() if v is False])

        print(
            f"Out of {len(long_tracks)}, {valid} / {valid + invalid} were annotated as correct. ({valid / (valid + invalid) * 100:.1f}%)"  # noqa: E501
        )

        metrics[method] = {
            "num_tracks": len(long_tracks),
            "valid_tracks": valid,
            "invalid_tracks": invalid,
            "score": valid / (valid + invalid),
        }

        # Update stored infos
        (experiment_path / f"valid_{method}.yml").write_text(yaml.dump(classification))
        (experiment_path / "metrics.yml").write_text(yaml.dump(metrics))

        print()
        print()

    print("-" * 100)
    print("Updated metrics:")
    print(yaml.safe_dump(metrics))


if __name__ == "__main__":
    main()
