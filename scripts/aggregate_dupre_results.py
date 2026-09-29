import pathlib

import numpy as np
import yaml


def main():  # noqa: C901, PLR0912
    detections = ["Fake@100%", "Fake@90%", "Fake@80%", "Fake@70%", "Fake@60%"]
    methods = ["trackmate-kf", "emht", "skt", "koft--", "koft"]
    results: dict[str, dict[str, list[float]]] = {
        method: {detection_name: [] for detection_name in detections} for method in methods
    }

    col_size = 20
    lines = []
    lines.append(
        "=" * ((col_size * len(detections)) // 2) + " " + "DUPRE" + " " + "=" * ((col_size * len(detections)) // 2)
    )
    lines.append("")
    lines.append("|".join([f"{'Method':15}"] + [f"{detection_name:^{col_size}}" for detection_name in detections]))
    lines.append("-" * ((len(detections) + 1) * col_size + len(detections) - (col_size - 15)))

    paths = (pathlib.Path("experiment_folder") / "tracking" / "dupre").glob("*/*")
    for path in paths:
        if not (path / "metrics.yml").exists():
            continue  # Run has not finished

        detection_cfg = yaml.safe_load((path / "config.yml").read_text())["detection"]

        detection_name = ""
        if detection_cfg["detector"] == "wavelet":
            detection_name = "Wavelet"
        else:
            if detection_cfg["fake"]["fpr"] == 0.0 and detection_cfg["fake"]["fnr"] == 0.0:
                detection_name = "Fake@100%"
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

        metrics = yaml.safe_load((path / "metrics.yml").read_text())
        for method in metrics:
            if method not in results:
                continue
            results[method][detection_name].append(metrics[method]["HOTA"])

    for method in methods:
        line = [f"{method:15}"]
        for detection_name in detections:
            scores = results[method][detection_name]
            if not scores:
                scores = [-1]
            mean, std = np.mean(scores), np.std(scores)
            line.append(f"{f'{mean * 100:.1f} +/- {std * 100:0.1f}% ({len(scores)})':>{col_size}}")

        lines.append("|".join(line))

    print("\n".join(lines))


if __name__ == "__main__":
    main()
