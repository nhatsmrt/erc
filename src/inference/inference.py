import torch
from torch import Tensor, nn
from nntoolbox.vision.models import ImageClassifier
from nntoolbox.utils import get_device
from torch.utils.data import Dataset
import numpy as np
from numpy import ndarray
import pandas as pd


class DummyTransform:
    def __call__(self, input): return input


class EmoRec(ImageClassifier):
    def __init__(self, model: nn.Module, shared_transform=DummyTransform(), tta_transform=None, device=get_device()):
        super().__init__(model, tta_transform, 0.0, device)
        self.shared_transform = shared_transform

    @torch.no_grad()
    def predict(self, image: Tensor, return_probs: bool=False, tries: int=5) -> ndarray:
        """
        Predict the classes or class probabilities of a batch of images

        :param image: image to be predicted
        :param return_probs: INVALID FOR THIS CLASS.
        :param tries: number of tries for augmentation
        :return:
        """
        if self._tta_transform is not None:
            version = [self._tta_transform(image) for _ in range(tries)]
            outputs = self._softmax(self._model(torch.stack(version, dim=0)))
            outputs = outputs.mean(dim=0).argmax(0, keepdim=True)
        else:
            outputs = self._model(image.unsqueeze(0)).argmax(1)

        return outputs.detach().cpu().numpy()

    @torch.no_grad()
    def export_predictions(self, test_data: Dataset, path: str, tries: int=5):
        outputs = []
        for image in test_data:
            image = self.shared_transform(image.to(self._device))
            outputs.append(self.predict(image, tries=tries))

        outputs = np.concatenate(outputs, axis=0)
        df_submission = pd.DataFrame({"File": test_data.filenames, "Label": outputs})
        df_submission.to_csv(path, index=False)
