from __future__ import annotations

import dataclasses
import enum
import pathlib

import byotrack
import byotrack.napari
import dacite
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

    video = cfg.data.video()

    # Detections
    detector = StarDistDetector.from_trained(cfg.stardist_model)
    detections_sequence = detector.run(video)

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

        byotrack.Track.save(tracks, f"tracks_{method.value}.pt")
        byotrack.Track.save(long_tracks, f"tracks_{method.value}.pt")

        print(f"{method.value} produced {len(tracks)} tracks with {len(long_tracks)} long trajectories (> 90% tracked)")

        viewer = byotrack.napari.visualize(video, detections_sequence)
        classification = add_track_classifier(viewer, long_tracks, max_distance=5.0)

        valid = len([k for k, v in classification.items() if v])

        print(f"Out of {len(long_tracks)}, {valid} were annotated as correct. ({valid / len(long_tracks) * 100:.1f}%)")

        metrics[method.value]["num_tracks"] = len(long_tracks)
        metrics[method.value]["valid_tracks"] = valid
        metrics[method.value]["score"] = valid / len(long_tracks)

    pathlib.Path("metrics.yml").write_text(yaml.dump(metrics))
