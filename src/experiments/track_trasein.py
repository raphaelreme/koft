from __future__ import annotations

import dataclasses
import enum
import pathlib

import byotrack
import byotrack.napari
import dacite
import napari  # type: ignore[import-untyped]
import numpy as np
import tqdm.auto as tqdm
import yaml
from byotrack.implementation.detector.stardist import StarDistDetector
from byotrack.implementation.refiner.interpolater import ForwardBackwardInterpolater

from src.experiments.track import ExperimentConfig as _ExperimentConfig
from src.manual_classification import add_track_classifier
from src.utils import enforce_all_seeds


@dataclasses.dataclass
class StardistConfig:
    stardist_model: pathlib.Path


@dataclasses.dataclass
class ExperimentConfig(_ExperimentConfig, StardistConfig):
    pass


def main(name: str, cfg_data: dict) -> None:
    print("Running:", name)
    print(yaml.dump(cfg_data))
    cfg = dacite.from_dict(ExperimentConfig, cfg_data, dacite.Config(cast=[pathlib.Path, tuple, enum.Enum]))

    enforce_all_seeds(cfg.seed)

    video = np.asarray(cfg.data.video())  # Video is long to decompress, let's do it once.

    # Detections
    detector = StarDistDetector.from_trained(cfg.stardist_model)
    detector.prob_threshold = 0.05
    if not (cfg.data.trasein.video.parent / f"detections-prob{detector.prob_threshold}").exists():
        detections_sequence = detector.run(video)
        byotrack.Detections.save_multi_frames_detections(
            detections_sequence, cfg.data.trasein.video.parent / f"detections-prob{detector.prob_threshold}"
        )
    else:
        detections_sequence = byotrack.Detections.load_multi_frames_detections(
            cfg.data.trasein.video.parent / f"detections-prob{detector.prob_threshold}"
        )

    byotrack.Detections.save_multi_frames_detections(detections_sequence, "detections")

    det_per_frame = sum(len(detections) for detections in detections_sequence) / len(detections_sequence)
    print(f"Found in average {det_per_frame} detections per frame")

    refiner = ForwardBackwardInterpolater()
    metrics: dict[str, dict[str, int | float]] = {}
    for method, linker in zip(
        cfg.tracking_methods, tqdm.tqdm(cfg.linkers(detections_sequence, video, [])), strict=True
    ):
        if hasattr(linker, "setup"):  # TrackOnStra
            linker.setup(video, detections_sequence)

        tracks = linker.run(video, detections_sequence)

        # Fill miss detected positions (if not done in the linker)
        tracks = refiner.run(video, tracks)

        # Let's keep long tracks than spans over 90% the video
        long_tracks = [track for track in tracks if len(track) > 0.9 * len(video)]

        tqdm.tqdm.write(
            f"{method.value} produced {len(tracks)} tracks with {len(long_tracks)} long trajectories (> 90% tracked)"
        )

        if not long_tracks:
            tqdm.tqdm.write("No long tracks. SKIPPED.")
            continue

        byotrack.Track.save(tracks, f"tracks_{method.value}.pt")
        byotrack.Track.save(long_tracks, f"long_tracks_{method.value}.pt")

        viewer = byotrack.napari.visualize(video, run=False)
        classification = add_track_classifier(viewer, long_tracks, max_distance=5.0)
        viewer._layer_slicer._force_sync = False  # noqa: SLF001  # Prevent lag
        viewer.layers[-1].visible = False  # May induce lag if visible
        napari.run()  # Modify in place `classification` from user inputs

        pathlib.Path(f"valid_{method.value}.yml").write_text(yaml.dump(classification))

        valid = len([k for k, v in classification.items() if v])

        tqdm.tqdm.write(
            f"Out of {len(long_tracks)}, {valid} were annotated as correct. ({valid / len(long_tracks) * 100:.1f}%)"
        )

        metrics[method.value] = {
            "num_tracks": len(long_tracks),
            "valid_tracks": valid,
            "score": valid / len(long_tracks),
        }

    pathlib.Path("metrics.yml").write_text(yaml.dump(metrics))
