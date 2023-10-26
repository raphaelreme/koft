from typing import Dict, Iterable, List, Optional

import torch

import byotrack
import pylapy


def make_increasing(points: Iterable[float]) -> List[float]:
    """Used to make monotone recalls and precision"""
    increasing = []
    max_point = 0.0
    for point in points:
        if point > max_point:
            max_point = point

        increasing.append(max_point)

    return increasing


def compute_ap(recalls: List[float], precisions: List[float]) -> float:
    """Compute average precision with AP = \\sum_k [R_{k+1} - R_k] * P_k

    Handle non monotone recall or precisions. (Recalls is globally increasing,
    precision is globally decreasing)

    Args:
        recalls (List[float]): Recall list for each level of cost limit
        precision (List[float]) Precision list for each level of cost limit

    Returns:
        float: Average precision metrics (precision is set to 0 for each point without recall)
    """
    recalls = make_increasing(recalls)
    precisions = make_increasing(reversed(precisions))
    precisions.reverse()

    # Extend the curve
    recalls = [0.0] + recalls + [max(recalls)]
    precisions = [max(precisions)] + precisions + [0.0]

    average_precision = 0.0
    for i, precision in enumerate(precisions[:-1]):
        average_precision += precision * (recalls[i + 1] - recalls[i])

    return average_precision


class DetectionMetric:
    """"""

    def __init__(self, dist_thresh: float, greedy=True) -> None:
        self.dist_thresh = dist_thresh
        self.greedy = greedy
        self.lap_solver = pylapy.LapSolver()

    def compute_at(
        self,
        detections: byotrack.Detections,
        true_position: torch.Tensor,
        true_weight: Optional[torch.Tensor] = None,
        prob_thresh=0.0,
        weight_thresh=0.0,
    ) -> Dict[str, float]:
        """Compute the precision, recall and f1 at a given probability and weight thresholds"""
        if true_weight is not None:
            true_position = true_position[true_weight > weight_thresh]

        predicted_position = detections.position[detections.confidence > prob_thresh]

        dist = torch.cdist(predicted_position, true_position)

        if self.greedy:
            dist[dist > self.dist_thresh] = torch.inf
            tp = self.lap_solver.solve(dist.numpy()).shape[0]
        else:
            tp = self.lap_solver.solve(dist.numpy(), self.dist_thresh).shape[0]

        n_pred = len(predicted_position)
        n_true = len(true_position)
        precision = tp / n_pred if n_pred else 1.0
        recall = tp / n_true if n_true else 1.0
        f1 = 2 * tp / (n_true + n_pred) if n_pred + n_true else 1.0

        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "n_pred": n_pred,
            "n_true": n_true,
            "tp": tp,
        }

    def average_precision_weight(
        self,
        detections: byotrack.Detections,
        true_position: torch.Tensor,
        true_weight: Optional[torch.Tensor] = None,
        prob_thresh=0.0,
    ) -> float:
        recalls = []
        precisions = []

        for weight_thresh in torch.linspace(0, 2.0, 201):
            metrics = self.compute_at(detections, true_position, true_weight, prob_thresh, weight_thresh.item())
            recalls.append(metrics["recall"])
            precisions.append(metrics["precision"])

        return compute_ap(recalls, precisions)

    def average_precision_prob(
        self,
        detections: byotrack.Detections,
        true_position: torch.Tensor,
        true_weight: Optional[torch.Tensor] = None,
        weight_thresh=0.0,
    ) -> float:
        recalls = []
        precisions = []

        for prob_thresh in torch.linspace(1.0, 0.0, 101):
            metrics = self.compute_at(detections, true_position, true_weight, prob_thresh.item(), weight_thresh)
            recalls.append(metrics["recall"])
            precisions.append(metrics["precision"])

        return compute_ap(recalls, precisions)
