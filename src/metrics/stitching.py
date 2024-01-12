"""Tracklet stitching metrics from Reme (ISBI_2023)"""

import numpy as np
import tqdm

import pylapy


def make_increasing(points):
    """Used to make monotone recalls and precision"""
    increasing = []
    max_point = 0.0
    for point in points:
        if point > max_point:
            max_point = point

        increasing.append(max_point)

    return increasing


def compute_ap(recalls, precisions) -> float:
    """Compute average precision with AP = \\sum_k [R_{k+1} - R_k] * P_k

    Will handle the fact that in our case recall is not monotone nor goes up to 1.

    Args:
        recalls (List[float]): Recall list for each level of cost limit
        precision (List[float]) Precision list for each level of cost limit

    Returns:
        float: Average precision metrics (precision is set to 0 for each point without recall)
    """
    # Let's put the first point as 1 of precision for 0 of recall by default (when predicting no links)
    recalls = [0.0] + recalls
    precisions = [1.0] + precisions

    average_precision = 0.0
    for i, precision in enumerate(precisions[:-1]):  # Drop last (anyway the recall diff is null)
        average_precision += precision * (recalls[i + 1] - recalls[i])

    return average_precision


def compute_metrics(dist, true_links):
    lap_solver = pylapy.LapSolver()

    results: dict = {
        "cost_limit": [],
        "recall": [],
        "precision": [],
        "prediction_proportion": [],
        "f1": [],
    }

    cost_limits = np.logspace(0, 1.5, 20)

    for cost_limit in tqdm.tqdm(cost_limits, "Solving with different cost limit"):
        links = lap_solver.solve(dist, eta=cost_limit)

        true_positives = 0

        for link in links:
            if tuple(link.tolist()) in true_links:
                true_positives += 1

        num_predictions = len(links)

        results["cost_limit"].append(float(cost_limit))
        results["recall"].append(true_positives / len(true_links))
        results["precision"].append(
            (true_positives + (num_predictions == 0)) / (num_predictions + (num_predictions == 0))
        )
        results["prediction_proportion"].append(num_predictions / dist.shape[0])
        results["f1"].append(
            2
            * results["precision"][-1]
            * results["recall"][-1]
            / (results["precision"][-1] + results["recall"][-1] + 1e-10)
        )

    # Make the recall monotone as it is usually (Useful for the computation of AP)
    results["monotone_recall"] = make_increasing(results["recall"])
    # Precision smoothing as done usually: the reverse precision is always going up
    results["monotone_precision"] = make_increasing(reversed(results["precision"]))
    results["monotone_precision"].reverse()

    results["average_precision"] = compute_ap(results["monotone_recall"], results["monotone_precision"])
    results["best_f1"] = max(results["f1"])
    results["best_recall"] = max(results["recall"])

    return results
