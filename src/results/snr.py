import pathlib

import numpy as np
import yaml

from src.results.drawing_helpers import plot_snr_impact


def main(simulation_name="springs_2d"):  # noqa: C901, PLR0912
    alphas = [0.02, 0.07, 0.2, 0.35, 0.85]
    deltas = [5.0, 50.0, 500.0]

    col_size = 20

    lines = []
    lines.append(
        "=" * ((col_size * len(alphas)) // 2)
        + " "
        + simulation_name.upper()
        + " "
        + "=" * ((col_size * len(alphas)) // 2)
    )
    lines.append("")
    lines.append("|".join([f"{'':10}"] + [f"{f'a={alpha}':^{col_size}}" for alpha in alphas]))
    lines.append("-" * ((len(alphas) + 1) * col_size + len(alphas) - (col_size - 10)))

    results: dict[float, dict[float, list[float]]] = {delta: {alpha: [] for alpha in alphas} for delta in deltas}

    paths = (pathlib.Path("experiment_folder") / "tracking_snr" / f"{simulation_name}").glob("*/*/*")
    for path in paths:
        if not (path / "metrics.yml").exists():
            continue  # Run has not finished

        alpha, delta = [float(k) for k in path.parent.parent.name.split("-")]
        if alpha not in alphas or delta not in deltas:
            continue

        detection_cfg = yaml.safe_load((path / "config.yml").read_text())["detection"]

        if detection_cfg["detector"] != "fake":
            continue
        if detection_cfg["fake"]["fpr"] != 0.2 or detection_cfg["fake"]["fnr"] != 0.2:
            continue

        metrics = yaml.safe_load((path / "metrics.yml").read_text())
        if "koft" not in metrics:
            continue

        results[delta][alpha].append(metrics["koft"]["HOTA"])

    for delta in deltas:
        line = [f"{f'd={delta}':<{10}}"]
        for alpha in alphas:
            scores = results[delta][alpha]
            if not scores:
                scores = [-1]
            mean, std = np.mean(scores), np.std(scores)
            line.append(f"{f'{mean * 100:.1f} +/- {std * 100:0.1f}% ({len(scores)})':>{col_size}}")

        lines.append("|".join(line))

    print("\n".join(lines))

    # Reload results for eMHT / TrackMate / SKT
    non_impacted_results: dict[str, list[float]] = {method: [] for method in ["skt", "trackmate-kf", "emht"]}
    paths = (pathlib.Path("experiment_folder") / "tracking" / f"{simulation_name}" / "0.2-50.0").glob("*/*")
    for path in paths:
        if not (path / "metrics.yml").exists():
            continue  # Run has not finished

        detection_cfg = yaml.safe_load((path / "config.yml").read_text())["detection"]

        # Select Fake@80%
        if detection_cfg["detector"] == "wavelet":
            continue
        if detection_cfg["fake"]["fpr"] != 0.2 or detection_cfg["fake"]["fnr"] != 0.2:
            continue

        metrics = yaml.safe_load((path / "metrics.yml").read_text())
        for method in non_impacted_results:  # noqa: PLC0206
            if method not in metrics:
                continue
            non_impacted_results[method].append(metrics[method]["HOTA"])

    plot_snr_impact(alphas, deltas, results, non_impacted_results, f"snr_track_{simulation_name}.svg")


if __name__ == "__main__":
    main("springs_2d")
    print()
    print()
    main("hydra_flow")
