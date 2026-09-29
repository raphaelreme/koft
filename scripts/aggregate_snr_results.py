import pathlib

import numpy as np
import yaml


def main(simulation_name="springs_2d"):
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


if __name__ == "__main__":
    main("springs_2d")
    print()
    print()
    main("hydra_flow")
