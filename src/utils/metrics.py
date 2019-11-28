from nntoolbox.metrics import Metric
from nntoolbox.callbacks import Callback
from typing import Dict, Any
from sklearn.metrics import accuracy_score, confusion_matrix
import torch


__all__ = ['ConfusionMatrixCB']


class ConfusionMatrixCB(Callback):
    def __init__(self, is_binary: bool=False):
        self.is_binary = is_binary

    def on_epoch_end(self, logs: Dict[str, Any]) -> bool:
        predictions = logs["outputs"]
        if self.is_binary:
            predictions = torch.round(predictions)[:, 0].cpu().detach().numpy()
        else:
            predictions = predictions.argmax(1).cpu().detach().numpy()

        labels = logs["labels"]
        if isinstance(labels, torch.Tensor):
            labels = labels.cpu().numpy()

        confusion_mat = confusion_matrix(y_true=labels, y_pred=predictions)

        print("Validation confusion matrix: ")
        print(confusion_mat)
        return False


class BinaryAccuracy(Metric):
    def __init__(self):
        self._best = 0.0

    def __call__(self, logs: Dict[str, Any]) -> float:
        if isinstance(logs["outputs"], torch.Tensor):
            predictions = torch.round(logs["outputs"])[:, 0].cpu().detach().numpy()
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
