import pathlib
import shutil

import byotrack.dataset.ctc
import tifffile
import torch
from byotrack.implementation.detector import wavelet

DATA_PATH = pathlib.Path(__file__).parent.parent.parent / "dataset"


def sinetra_to_ctc(sinetra_path: pathlib.Path, output_dir: pathlib.Path) -> None:
    """Convert a sinetra dataset to the CTC format."""
    seed = sinetra_path.name
    specs = sinetra_path.parent.name
    motion = sinetra_path.parent.parent.name

    dataset_path = output_dir / motion / specs
    dataset_path.mkdir(parents=True, exist_ok=True)

    # Let's copy the video
    video_path = dataset_path / seed

    if video_path.exists():
        shutil.rmtree(video_path)

    video_path.mkdir()

    video = byotrack.Video(sinetra_path / "video.tiff")
    for i, frame in enumerate(video):
        tifffile.imwrite(video_path / f"{i:04}.tif", frame[None, None, None], imagej=True, compression="zlib")

    # Let's copy the GT tracks
    gt_path = dataset_path / f"{seed}_GT" / "TRA"
    if gt_path.exists():
        shutil.rmtree(gt_path)

    shutil.copytree(sinetra_path / "tracks", gt_path)
    for path in gt_path.glob("*.tiff"):
        path.rename(path.with_suffix(".tif"))

    # Finally, let's create the man_track.txt
    n_track = torch.load(sinetra_path / "video_data.pt")["mu"].shape[1]
    metadata = "\n".join(f"{i + 1} 0 199 0" for i in range(n_track))
    (gt_path / "man_track.txt").write_text(metadata)


def convert_with_segmentations(motion: str, seed: int) -> None:
    path = DATA_PATH / f"{motion}" / "0.2-50.0" / f"{seed}"

    sinetra_to_ctc(path, DATA_PATH / "trackastra")

    video = byotrack.Video(path / "video.tiff").normalize()

    # Run wavelet segmentation
    detector = wavelet.WaveletDetector(scale=1, k=2.5, min_area=8.0)
    detections_sequence = detector.run(video)

    byotrack.dataset.ctc.save_detections(
        detections_sequence, DATA_PATH / "trackastra" / f"{motion}" / "0.2-50.0" / f"{seed}_ERR_SEG"
    )


if __name__ == "__main__":
    # convert_with_segmentations("springs_2d", 11)
    # convert_with_segmentations("springs_2d", 22)
    convert_with_segmentations("hydra_flow", 33)
    convert_with_segmentations("hydra_flow", 44)
