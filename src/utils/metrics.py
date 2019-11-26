from nntoolbox.metrics import Metric
from typing import Dict, Any
from sklearn.metrics import accuracy_score
import torch


class BinaryAccuracy(Metric):
    def __init__(self):
        self._best = 0.0

    def __call__(self, logs: Dict[str, Any]) -> float:
        if isinstance(logs["outputs"], torch.Tensor):
            predictions = torch.round(logs["outputs"]).cpu().detach().numpy()
        else:
            predictions = logs["outputs"]

        labels = logs["labels"]
        if isinstance(labels, torch.Tensor):
            labels = labels.cpu().numpy()

        acc = accuracy_score(
            y_true=labels.ravel(),
            y_pred=predictions.ravel()
        )

        if acc >= self._best:
            self._best = acc

        return acc
