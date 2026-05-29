import pathlib

import numpy as np
import yaml


def main():  # noqa: C901
    print(f"{'method':15}|{'Fake@90%':20}|{'Fake@80%':20}|{'Fake@70%':20}|{'Fake@60%':20}")
    detections = ["Fake@90%", "Fake@80%", "Fake@70%", "Fake@60%"]
    for method in ["trackmate", "trackmate-kf", "emht", "skt", "koft--", "koft", "koft++"]:
        aggregated = []
        paths = pathlib.Path.glob(pathlib.Path("experiments_folder") / "tracking" / "dupre" / f"{method}", "*")
        results: dict[str, list[float]] = {detection_name: [] for detection_name in detections}

        for path in paths:
            if not (path / "best_metrics.yml").exists():
                continue  # Run has not finished

            detection_cfg = yaml.safe_load((path / "config.yml").read_text())["detection"]

            detection_name = ""
            if detection_cfg["fake"]["fpr"] == 0.1 and detection_cfg["fake"]["fnr"] == 0.1:
                detection_name = "Fake@90%"
            if detection_cfg["fake"]["fpr"] == 0.2 and detection_cfg["fake"]["fnr"] == 0.2:
                detection_name = "Fake@80%"
            if detection_cfg["fake"]["fpr"] == 0.3 and detection_cfg["fake"]["fnr"] == 0.3:
                detection_name = "Fake@70%"
            elif detection_cfg["fake"]["fpr"] == 0.4 and detection_cfg["fake"]["fnr"] == 0.4:
                detection_name = "Fake@60%"

            if not detection_name:
                continue

            hota = yaml.safe_load((path / "config.yml").read_text())["HOTA"]

            results[detection_name].append(hota)

        for detection_name in detections:
            scores = results[detection_name]
            if not scores:
                scores = [-1]
            mean, std = np.mean(scores), np.std(scores)
            aggregated.append(f"{mean * 100:.1f} +/- {std * 100:0.1f}% ({len(scores)})")

        print(f"{method:15}|{aggregated[0]:20}|{aggregated[1]:20}|{aggregated[2]:20}|{aggregated[3]:20}")


if __name__ == "__main__":
    main()
